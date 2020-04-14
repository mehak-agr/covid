
echo '1'
CUDA_VISIBLE_DEVICES=3 python3 -u -m medinet.demo_chestxray --data './data/' --resume '/data/2015P002510/Mehak/classification-covid/model_debug/lr0.0001_lrp0.1_adam_epochs3_chex_baseline_densenet_unbalanced/checkpoint.pth.tar' --model-dir './models/' --lr 0.0001 --epochs 30 --balanced 0 --train_csv './images/chexpert/train_chexpert_uncur.csv' --val_csv './images/chexpert/valid_chexpert_uncur.csv' | tee -a './outxt/result.txt'

# Arguments
#  '--data', help='path to dataset')
#  '--model-dir', default='./models/', type=str, metavar='MODELPATH', help='path to model directory (default: none)')
#  '--image-size', '-i', default=320, type=int, metavar='N', help='image size (default: 320)')
#  '-j', '--workers', default=4, type=int, metavar='N', help='number of data loading workers (default: 4)')
#  '--epochs', default=20, type=int, metavar='N', help='number of total epochs to run')
#  '--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
#  '-b', '--batch_size', default=16, type=int, metavar='N', help='mini-batch size (default: 16)')
#  '--lr', '--learning-rate', default=0.0001, type=float, metavar='LR', help='initial learning rate')
#  '--lrp', '--learning-rate-pretrained', default=0.1, type=float, metavar='LR', help='learning rate for pre-trained layers')
#  '--momentum', default=0.9, type=float, metavar='M', help='momentum')
#  '--weight-decay', '--wd', default=1e-4, type=float, metavar='W', help='weight decay (default: 1e-4)')
#  '--print_freq', '-p', default=0, type=int, metavar='N', help='print frequency (default: 0)')
#  '--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
#  '-e', '--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')
#  '--k', default=0, type=float, metavar='N', help='number of regions (default: 0)')
#  '--alpha', default=0, type=float, metavar='N', help='weight for the min regions (default: 0)')
#  '--maps', default=0, type=int, metavar='N', help='number of maps per class (default: 0)')
#  '--adam', default=1, type=int, metavar='A', help='Use Adam')
#  '--arch', default=0, type=int, metavar='w', help='Use Baseline/Wildcat/Weldon')
#  '--variant', default=0, type=int, metavar='w', help='Use Densenet/Resnet/VGG')
#  '--train_csv', default='train.csv', type=str, metavar='r', help='Give train csv')
#  '--val_csv', default='val.csv', type=str, help='Give validation csv')
#  '--sigmoid', default='1', type=int, help='Specify if you need sigmoid activation in baseline')
#  '--balanced', default='1', type=int, help='Specify if you need balanced sampling or not (Default:BalancedSampling)')
#  '--loss', default='0', type=int, help='Specify which criterion you need (default:BCELoss)')

