import tensorflow as tf
import numpy as np
import cv2
from utils import box_iou
from data import dataGenerator, preprocess_true_boxes

class YOLOv3:
    def __init__(
            self,
            anchors_file,
            num_classes,
            train_file,
            epochs,
            batch_size,
            lr,
            lr_decay,
            shuffle,
            repeat,
            snapshots
                 ):
        self.anchors = self.get_anchors(anchors_file)
        self.num_classes = num_classes
        self.train_file = train_file
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.lr_deacy = lr_decay
        self.shuffle = shuffle
        self.repeat = repeat
        self.snapshots = snapshots


    def get_anchors(self, anchors_path):
        '''loads the anchors from a file'''
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    def conv_block(self, inputs, num_filters, kernel_size, strides=(1,1), padding='same'):
        x = tf.keras.layers.Conv2D(num_filters, kernel_size, strides, padding)(inputs)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)

        return x

    def res_block(self, inputs, num_filters, num_blocks):
        x = self.conv_block(inputs, num_filters, (3, 3), strides=(2, 2))
        for i in range(num_blocks):
            y = self.conv_block(x, num_filters // 2, (1, 1))
            y = self.conv_block(y, num_filters, (3, 3))

            x = tf.keras.layers.Add()([x, y])

        return x

    def backbone(self, inputs):
        x = self.conv_block(inputs, 32, (3,3))
        x = self.res_block(x, 64, 1)
        x = self.res_block(x, 128, 2)
        x = self.res_block(x, 256, 8)
        x = self.res_block(x, 512, 8)
        x = self.res_block(x, 1024, 4)

        return x

    def generate_predict_layers(self, inputs, num_filters, out_filters):
        x = self.conv_block(inputs, num_filters, (1,1))
        x = self.conv_block(x, num_filters*2, (3,3))
        x = self.conv_block(x, num_filters, (1,1))
        x = self.conv_block(x, num_filters*2, (3,3))
        x = self.conv_block(x, num_filters, (1,1))

        y = self.conv_block(x, num_filters*2, (3,3))
        y = tf.keras.layers.Conv2D(out_filters, (1,1))(y)

        return x, y

    def yolo_outputs(self, inputs, num_anchors, num_classes):
        darknet53 = tf.keras.Model(inputs, self.backbone(inputs))

        x, y1 = self.generate_predict_layers(darknet53.output, 512, num_anchors*(num_classes+5))

        x = self.conv_block(x, 256, (1,1))
        x = tf.keras.layers.UpSampling2D(2)(x)
        x = tf.keras.layers.Concatenate()([x, darknet53.layers[147].output])
        x, y2 = self.generate_predict_layers(x, 256, num_anchors*(num_classes+5))

        x = self.conv_block(x, 128, (1, 1))
        x = tf.keras.layers.UpSampling2D(2)(x)
        x = tf.keras.layers.Concatenate()([x, darknet53.layers[87].output])
        x, y3 = self.generate_predict_layers(x, 128, num_anchors * (num_classes + 5))

        return [y1, y2, y3]

    def yolo_head(self, features, anchors, num_classes, input_shape, calc_loss=False):
        num_anchors = len(anchors)

        anchors_tensor = tf.reshape(tf.constant(anchors), [1, 1, 1, num_anchors, 2])

        grid_shape = features.shape[1:3]

        grid_y = tf.tile(tf.reshape(
            tf.keras.backend.arange(0, stop=grid_shape[0]),
            [-1, 1, 1, 1]), [1, grid_shape[1], 1, 1])
        grid_x = tf.tile(tf.reshape(
            tf.keras.backend.arange(0, stop=grid_shape[1]),
            [1, -1, 1, 1]), [grid_shape[0], 1, 1, 1])
        grid = tf.keras.layers.concatenate([grid_x, grid_y])
        grid = tf.cast(grid, features.dtype)

        features = tf.reshape(features, [self.batch_size, grid_shape[0], grid_shape[1], num_anchors, num_classes + 5])

        box_xy = (tf.sigmoid(features[..., :2]) + grid) / tf.cast(grid_shape[::-1], features.dtype)
        box_wh = tf.exp(features[...,2:4]) * tf.cast(anchors_tensor, features.dtype) / tf.cast(input_shape[::-1], features.dtype)
        box_confidence = tf.sigmoid(features[..., 4:5])
        box_class_probs = tf.sigmoid(features[..., 5:])

        if calc_loss == True:
            return grid, features, box_xy, box_wh

        return box_xy, box_wh, box_confidence, box_class_probs

    def loss(self, y_true, y_pred, ignore_thresh):
        anchors = self.anchors
        num_classes = 10

        num_anchors, num_layers = len(anchors) // 3, len(anchors) // 3
        anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]] if num_layers == 3 else [[3, 4, 5], [1, 2, 3]]
        input_shape = tf.cast(y_pred[0].get_shape()[1:3] * tf.constant(32), y_pred[0].dtype)
        grid_shape = [tf.cast(y_pred[i].get_shape()[1:3], y_pred[0].dtype) for i in range(num_layers)]
        loss = 0
        batch_size = self.batch_size

        for i in range(num_layers):
            object_mask = tf.convert_to_tensor(y_true[i][..., 4:5])
            cls_confidence_true = tf.convert_to_tensor(y_true[i][..., 5:])

            grid, raw_pred, pred_xy, pred_wh = self.yolo_head(y_pred[i], anchors[anchor_mask[i]], num_classes, input_shape, calc_loss=True)
            pred_box = tf.keras.layers.concatenate([pred_xy, pred_wh])

            true_t_xy = y_true[i][..., :2] * grid_shape[i][::-1] - grid
            true_t_wh = tf.math.log(y_true[i][..., 2:4] / anchors[anchor_mask[i]] * input_shape[::-1])
            true_t_wh = tf.keras.backend.switch(object_mask, true_t_wh, tf.zeros_like(true_t_wh))  # avoid log(0)=-inf
            box_loss_scale = 2 - y_true[i][..., 2:3] * y_true[i][..., 3:4]

            # Find ignore mask, iterate over each of batch.
            ignore_mask = tf.TensorArray(y_true[0].dtype, size=1, dynamic_size=True)
            object_mask_bool = tf.cast(object_mask, 'bool')

            def loop_body(b, ignore_mask):
                true_box = tf.boolean_mask(y_true[i][b, ..., 0:4], object_mask_bool[b, ..., 0])
                iou = box_iou(pred_box[b], true_box)
                best_iou = tf.keras.backend.max(iou, axis=-1)
                ignore_mask = ignore_mask.write(b, tf.cast(best_iou < ignore_thresh, true_box.dtype))
                return b + 1, ignore_mask

            _, ignore_mask = tf.while_loop(lambda b, *args: b < batch_size, loop_body, [0, ignore_mask])
            ignore_mask = ignore_mask.stack()
            ignore_mask = tf.expand_dims(ignore_mask, -1)

            xy_loss = object_mask[..., 0] * box_loss_scale[..., 0] * tf.losses.binary_crossentropy(true_t_xy, raw_pred[..., 0:2], from_logits=True)
            wh_loss = object_mask * box_loss_scale * tf.square(true_t_wh - raw_pred[..., 2:4])
            confidence_loss = object_mask[..., 0] * tf.losses.binary_crossentropy(object_mask, raw_pred[..., 4:5], from_logits=True) + \
                              ignore_mask[..., 0] * (1 - object_mask[..., 0]) * tf.losses.binary_crossentropy(object_mask, raw_pred[..., 4:5], from_logits=True)
            cls_loss = object_mask[..., 0] * tf.losses.binary_crossentropy(cls_confidence_true, raw_pred[..., 5:], from_logits=True)

            xy_loss = tf.reduce_sum(xy_loss) / batch_size
            wh_loss = tf.reduce_sum(wh_loss) / batch_size
            confidence_loss = tf.reduce_sum(confidence_loss) / batch_size
            cls_loss = tf.reduce_sum(cls_loss) / batch_size

            loss = loss + xy_loss + wh_loss + confidence_loss + cls_loss

        return loss

    @tf.function
    def train_step(self, optimizer, model, images, labels):
        with tf.GradientTape() as tape:
            y_pred = model(images)
            loss = self.loss(y_pred=y_pred, y_true=labels, ignore_thresh=.5)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        tf.print('Total loss', loss)

    def train(self):
        inputs = tf.keras.Input([416, 416, 3])
        outputs = self.yolo_outputs(inputs, num_anchors=3, num_classes=self.num_classes)

        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        optimizer = tf.keras.optimizers.Adam(self.lr, self.lr_deacy)

        train_datasets = dataGenerator(
            annotation_file=self.train_file,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            repeat=self.repeat,
            mode='training'
        )

        for i in range(self.epochs):
            print('Epoch %d' % (i))
            for images, labels in train_datasets:

                true_boxes = np.ones((self.num_classes, 9, 5))
                for j in range(9):
                    true_boxes[:,j,:] = labels
                true_boxes = preprocess_true_boxes(
                    true_boxes,
                    anchors=self.anchors,
                    input_shape=[416, 416],
                    num_classes=self.num_classes
                )

                batch_images = []
                for item in images:
                    batch_images.append(cv2.imread(item, 1) / 255.0)
                batch_images = np.array(batch_images)

                self.train_step(
                optimizer,
                model,
                images=batch_images,
                labels=true_boxes)

            model.save(self.snapshots + 'epoch-' + str(i) + '.h5')

    def yolo_correct_boxes(self, box_xy, box_wh, input_shape, image_shape):
        '''Get corrected boxes'''
        box_yx = box_xy[..., ::-1]
        box_hw = box_wh[..., ::-1]
        input_shape = tf.cast(input_shape, box_yx.dtype)
        image_shape = tf.cast(image_shape, box_yx.dtype)
        new_shape = tf.round(image_shape * tf.keras.backend.min(input_shape / image_shape))
        offset = (input_shape - new_shape) / 2. / input_shape
        scale = input_shape / new_shape
        box_yx = (box_yx - offset) * scale
        box_hw *= scale

        box_mins = box_yx - (box_hw / 2.)
        box_maxes = box_yx + (box_hw / 2.)
        boxes = tf.keras.backend.concatenate([
            box_mins[..., 0:1],  # y_min
            box_mins[..., 1:2],  # x_min
            box_maxes[..., 0:1],  # y_max
            box_maxes[..., 1:2]  # x_max
        ])

        # Scale boxes back to original image shape.
        boxes *= tf.keras.backend.concatenate([image_shape, image_shape])
        return boxes

    def yolo_boxes_and_scores(self, feats, anchors, num_classes, input_shape, image_shape):
        '''Process Conv layer output'''
        box_xy, box_wh, box_confidence, box_class_probs = self.yolo_head(feats,
                                                                    anchors, num_classes, input_shape)
        boxes = self.yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape)
        boxes = tf.reshape(boxes, [-1, 4])
        box_scores = box_confidence * box_class_probs
        box_scores = tf.reshape(box_scores, [-1, num_classes])
        return boxes, box_scores

    def inference(self,
                  yolo_outputs,
                  image_shape,
                  max_boxes=20,
                  score_threshold=.6,
                  iou_threshold=.5):
        """Evaluate YOLO model on given input and return filtered boxes."""
        anchors = self.anchors
        num_classes = self.num_classes
        num_layers = len(yolo_outputs)
        anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]] if num_layers == 3 else [[3, 4, 5],
                                                                                 [1, 2, 3]]  # default setting
        input_shape = yolo_outputs[0].shape[1:3] * tf.constant(32)
        boxes = []
        box_scores = []
        for l in range(num_layers):
            _boxes, _box_scores = self.yolo_boxes_and_scores(yolo_outputs[l],
                                                        anchors[anchor_mask[l]], num_classes, input_shape, image_shape)
            boxes.append(_boxes)
            box_scores.append(_box_scores)
        boxes = tf.keras.backend.concatenate(boxes, axis=0)
        box_scores = tf.keras.backend.concatenate(box_scores, axis=0)

        mask = box_scores >= score_threshold
        max_boxes_tensor = tf.keras.backend.constant(max_boxes, dtype='int32')
        boxes_ = []
        scores_ = []
        classes_ = []
        for c in range(num_classes):
            # TODO: use keras backend instead of tf.
            class_boxes = tf.boolean_mask(boxes, mask[:, c])
            class_box_scores = tf.boolean_mask(box_scores[:, c], mask[:, c])
            nms_index = tf.image.non_max_suppression(
                class_boxes, class_box_scores, max_boxes_tensor, iou_threshold=iou_threshold)
            class_boxes = tf.keras.backend.gather(class_boxes, nms_index)
            class_box_scores = tf.keras.backend.gather(class_box_scores, nms_index)
            classes = tf.keras.backend.ones_like(class_box_scores, 'int32') * c
            boxes_.append(class_boxes)
            scores_.append(class_box_scores)
            classes_.append(classes)
        boxes_ = tf.keras.backend.concatenate(boxes_, axis=0)
        scores_ = tf.keras.backend.concatenate(scores_, axis=0)
        classes_ = tf.keras.backend.concatenate(classes_, axis=0)

        return boxes_, scores_, classes_

