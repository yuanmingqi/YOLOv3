import argparse
from model import YOLOv3

parser = argparse.ArgumentParser("Training options for the YOLOv3 model.")

parser.add_argument("--anchors_file", type=str, default='./data/anchors.txt')
parser.add_argument("--num_classes", type=int, default=1000)
parser.add_argument("--train_file", type=str, default='./data/train.npz')
parser.add_argument("--epochs", type=int, default=50)
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--lr_decay", type=float, default=0.0001)
parser.add_argument("--shuffle", type=bool, default=True)
parser.add_argument("--repeat", type=int, default=100)
parser.add_argument("--snapshots", type=str, default='./snapshots/')

args = parser.parse_args()

yolov3 = YOLOv3(
    anchors_file=args.anchors_file,
    num_classes=args.num_classes,
    train_file=args.train_file,
    epochs=args.epochs,
    batch_size=args.batch_size,
    lr=args.lr,
    lr_decay=args.lr_decay,
    shuffle=args.shuffle,
    repeat=args.repeat,
    snapshots=args.snapshots
)

if __name__ == 'main':
    yolov3.train()