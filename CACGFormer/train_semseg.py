import argparse
import os
import torch
import datetime
import logging
from pathlib import Path
import sys
import importlib
import shutil
from tqdm import tqdm
import provider
import numpy as np
import time

from sklearn.metrics import precision_score, recall_score, f1_score, balanced_accuracy_score
# Set base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

# Classes for training and testing datasets
classes = ['yayin', 'yaguan']
class2label = {cls: i for i, cls in enumerate(classes)}
seg_classes = class2label
seg_label_to_cat = {}
for i, cat in enumerate(seg_classes.keys()):
    seg_label_to_cat[i] = cat

def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True

class YachiDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, split='train', num_points=4096):
        self.root_dir = root_dir
        self.split = split
        self.num_points = num_points

        if self.split == 'train':
            self.files = [os.path.join(root_dir, f'yachi_yachi_{i}.npy') for i in range(1, 187)]  # Modify this based on your dataset187
        elif self.split == 'test':
            self.files = sorted([os.path.join(root_dir, f'Test_yachi_{i}.npy') for i in range(1, 53)])  # Modify this based on your dataset27
        else:
            raise ValueError("Split must be either 'train' or 'test'")
            
        self.labelweights = self.calculate_labelweights()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        data = np.load(self.files[index])
        points = data[:, :6]  # Extract XYZ and RGB
        labels = data[:, 6]  # Extract labels

        labels = np.where(labels == 0, 0, 1)  # Map 'yayin' to 0, 'yaguan' to 1

        if points.shape[0] > self.num_points:
            choice = np.random.choice(points.shape[0], self.num_points, replace=False)
            points = points[choice, :]
            labels = labels[choice]
        elif points.shape[0] < self.num_points:
            choice = np.random.choice(points.shape[0], self.num_points, replace=True)
            points = points[choice, :]
            labels = labels[choice]

        return points, labels

    def calculate_labelweights(self):
        label_count = np.zeros(2)
        for file in self.files:
            data = np.load(file)
            labels = data[:, 6]
            unique, counts = np.unique(labels, return_counts=True)
            for u, c in zip(unique, counts):
                label_count[int(u)] += c 
        
        weights = np.sum(label_count) / (label_count + 1e-6)
        weights = weights / np.sum(weights)  
        return weights

# Parse command line arguments
def parse_args():
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--model', type=str, default='pointnet2_sem_seg')
    parser.add_argument('--batch_size', type=int, default=6)
    parser.add_argument('--epoch', default=640, type=int)
    parser.add_argument('--learning_rate', default=0.005, type=float)
    parser.add_argument('--gpu', type=str, default='1')
    parser.add_argument('--optimizer', type=str, default='Adam')
    parser.add_argument('--log_dir', type=str, default=None)
    parser.add_argument('--decay_rate', type=float, default=5e-3)
    parser.add_argument('--npoint', type=int, default=32768)
    parser.add_argument('--step_size', type=int, default=10)
    parser.add_argument('--lr_decay', type=float, default=0.5)
    return parser.parse_args()
    
def main(args):
    def log_string(str):
        logger.info(str)
        print(str)
    
    best_metrics = {
        'mIoU': 0.0,
        'mAcc': 0.0,
        'allAcc': 0.0,
        'mPrecision': 0.0,
        'mRecall': 0.0,
        'class_ious': [],
        'class_precision': [],
        'class_recall': [],
    }

    '''HYPER PARAMETERS'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    '''CREATE DIRECTORY'''
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    experiment_dir = Path('./log/')
    experiment_dir.mkdir(exist_ok=True)
    experiment_dir = experiment_dir.joinpath('sem_seg')
    experiment_dir.mkdir(exist_ok=True)
    if args.log_dir is None:
        experiment_dir = experiment_dir.joinpath(timestr)
    else:
        experiment_dir = experiment_dir.joinpath(args.log_dir)
    experiment_dir.mkdir(exist_ok=True)
    checkpoints_dir = experiment_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = experiment_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)

    '''LOGGING CONFIGURATION'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETERS ...')
    log_string(args)

    root = '/data/zuowenhao/pointnet2_my/data/stanford_indoor3d/'  # Path to your data folder
    NUM_CLASSES = 2  # 'yayin' and 'yaguan'
    NUM_POINT = args.npoint
    BATCH_SIZE = args.batch_size

    # Load training data using YachiDataset
    print("Start loading training data...") 
    TRAIN_DATASET = YachiDataset(root_dir=root, split='train', num_points=NUM_POINT)
    
    # Load test data using YachiDataset
    print("Start loading test data...") 
    TEST_DATASET = YachiDataset(root_dir=root, split='test', num_points=NUM_POINT)

    # Create DataLoader for training and testing
    trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=BATCH_SIZE, shuffle=True, num_workers=16, pin_memory=True, drop_last=True)
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=1, shuffle=False, num_workers=16, pin_memory=True, drop_last=True)
    weights = torch.tensor(TRAIN_DATASET.labelweights).float().cuda()
    
    # Log the number of training and test data
    log_string("Number of training data: %d" % len(TRAIN_DATASET))
    log_string("Number of test data: %d" % len(TEST_DATASET))

    '''MODEL LOADING'''
    MODEL = importlib.import_module(args.model)
    shutil.copy('models/%s.py' % args.model, str(experiment_dir))
    shutil.copy('models/pointnet2_utils.py', str(experiment_dir))

    classifier = MODEL.get_model(NUM_CLASSES).cuda()
    criterion = MODEL.get_loss().cuda()

    classifier.apply(inplace_relu)

    # Initialize weights for the model
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv2d') != -1:
            torch.nn.init.xavier_normal_(m.weight.data)
            torch.nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('Linear') != -1:
            torch.nn.init.xavier_normal_(m.weight.data)
            torch.nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm') != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)
    # Try to load the pre-trained model checkpoint
    try:
        checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')
        start_epoch = checkpoint['epoch']
        classifier.load_state_dict(checkpoint['model_state_dict'])
        log_string('Using pre-trained model')
    except:
        log_string('No pre-trained model, starting training from scratch...')
        start_epoch = 0
        classifier = classifier.apply(weights_init)

    # Set optimizer
    optimizer = torch.optim.Adam(
        classifier.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=args.decay_rate
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=args.epoch, 
        eta_min=1e-5
    )

    '''TRAINING LOOP'''
    global_epoch = 0
    best_iou = 0
    total_correct = 0
    total_seen = 0

    for epoch in range(start_epoch, args.epoch):
        total_correct = 0
        total_seen = 0
        log_string(f'**** Epoch {global_epoch + 1} ({epoch + 1}/{args.epoch}) ****')
        lr = max(args.learning_rate * (args.lr_decay ** (epoch // args.step_size)), 1e-5)
        log_string(f'Learning rate: {lr}')
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        classifier = classifier.train()

        loss_sum = 0  # Initialize loss sum
        num_batches = len(trainDataLoader)  # Get the number of batches
        
        # Training loop
        for i, data in tqdm(enumerate(trainDataLoader), total=len(trainDataLoader), smoothing=0.9):
            points, target = data
            points = points.data.numpy()
            points[:, :, :3] = provider.rotate_point_cloud_z(points[:, :, :3])
            points[:, :, :3] = provider.random_scale_point_cloud(points[:, :, :3])
            points[:, :, :3] = provider.jitter_point_cloud(points[:, :, :3])
            
            points = torch.Tensor(points)
            points, target = points.float().cuda(), target.long().cuda()
            points = points.transpose(2, 1)
            optimizer.zero_grad()
            classifier = classifier.train()
            seg_pred, trans_feat = classifier(points)
            seg_pred = seg_pred.contiguous().view(-1, NUM_CLASSES)
            batch_label = target.view(-1, 1)[:, 0].cpu().data.numpy()
            target = target.view(-1, 1)[:, 0]
            loss = criterion(seg_pred, target, trans_feat, weights)  # Using the computed weights
            loss.backward()
            optimizer.step()
            scheduler.step()
            # Accumulate the loss
            loss_sum += loss.item()
            pred_choice = seg_pred.max(1)[1].cpu().data.numpy()
            correct = np.sum(pred_choice == batch_label)
            total_correct += correct
            total_seen += (BATCH_SIZE * NUM_POINT)

        log_string(f'Training average loss: {loss_sum / num_batches}')
        log_string(f'Training accuracy: {total_correct / float(total_seen)}')

        # Save model every 5 epochs
        if epoch % 5 == 0:
            savepath = str(checkpoints_dir) + '/model.pth'
            log_string(f'Saving model to {savepath}')
            state = {
                'epoch': epoch,
                'model_state_dict': classifier.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            torch.save(state, savepath)
            log_string('Saving model....')

        '''Evaluate on chopped scenes'''
        with torch.no_grad():
            num_batches = len(testDataLoader)
            total_correct = 0
            total_seen = 0
            loss_sum = 0
            labelweights = np.zeros(NUM_CLASSES)
            total_seen_class = [0 for _ in range(NUM_CLASSES)]
            total_correct_class = [0 for _ in range(NUM_CLASSES)]
            total_iou_deno_class = [0 for _ in range(NUM_CLASSES)]
            classifier = classifier.eval()
            all_labels = []  
            all_predictions = []  
            output_dir = '/data/zuowenhao/pointnet2_my/log/sem_seg/pointnet2_sem_seg/test_label/'
            os.makedirs(output_dir, exist_ok=True)
            
            log_string('---- EPOCH %03d EVALUATION ----' % (global_epoch + 1))
            for i, (points, target) in tqdm(enumerate(testDataLoader), total=len(testDataLoader), smoothing=0.9):
                points = points.data.numpy()
                points = torch.Tensor(points)
                points, target = points.float().cuda(), target.long().cuda()
                points = points.transpose(2, 1)
        
                seg_pred, trans_feat = classifier(points)
                pred_val = seg_pred.contiguous().cpu().data.numpy()
                seg_pred = seg_pred.contiguous().view(-1, NUM_CLASSES)
        
                batch_label = target.cpu().data.numpy()
                target = target.view(-1, 1)[:, 0]
                loss = criterion(seg_pred, target, trans_feat, weights)
                loss_sum += loss.item()
                pred_val = np.argmax(pred_val, 2)
                flat_labels = batch_label.flatten()
                flat_preds = pred_val.flatten()
                all_labels.extend(flat_labels)
                all_predictions.extend(flat_preds)
   
                correct = np.sum((pred_val == batch_label))
                total_correct += correct
                total_seen += (BATCH_SIZE * NUM_POINT)
                tmp, _ = np.histogram(batch_label, range(NUM_CLASSES + 1))
                labelweights += tmp

                for l in range(NUM_CLASSES):
                    total_seen_class[l] += np.sum((batch_label == l))
                    total_correct_class[l] += np.sum((pred_val == l) & (batch_label == l))
                    total_iou_deno_class[l] += np.sum(((pred_val == l) | (batch_label == l)))
                original_filename = TEST_DATASET.files[i]
    
                save_name = os.path.basename(original_filename).replace("Test_yachi", "test")
                np.save(os.path.join(output_dir, save_name), pred_val)
                
           
            all_labels = np.array(all_labels)
            all_predictions = np.array(all_predictions)
            
         
            balanced_acc = balanced_accuracy_score(all_labels, all_predictions)
            log_string(f'Balanced Accuracy: {balanced_acc:.4f}')
                
            precision_per_class = precision_score(all_labels, all_predictions, average=None, zero_division=0,labels=np.arange(NUM_CLASSES) )
            recall_per_class = recall_score(all_labels, all_predictions, average=None, zero_division=0,labels=np.arange(NUM_CLASSES))
            labelweights = labelweights.astype(np.float32) / np.sum(labelweights.astype(np.float32))
            iou_per_class = []
            for l in range(NUM_CLASSES):
                iou = total_correct_class[l] / (total_iou_deno_class[l] + 1e-6)
                iou_per_class.append(iou)
            mIoU = np.mean(iou_per_class)
            
            precision = precision_score(all_labels, all_predictions, average='macro', zero_division=0)
            recall = recall_score(all_labels, all_predictions, average='macro', zero_division=0)
            f1 = f1_score(all_labels, all_predictions, average='macro', zero_division=0)

            if mIoU >= best_iou:
                best_iou = mIoU
                best_metrics['mIoU'] = mIoU
                best_metrics['mAcc'] = np.mean(np.array(total_correct_class) / (np.array(total_seen_class, dtype=float) + 1e-6))
                best_metrics['allAcc'] = total_correct / float(total_seen)
                best_metrics['mPrecision'] = precision
                best_metrics['mRecall'] = recall
                best_metrics['class_ious'] = [total_correct_class[l] / float(total_iou_deno_class[l]) for l in range(NUM_CLASSES)]
                best_metrics['class_precision'] = precision_per_class
                best_metrics['class_recall'] = recall_per_class
                logger.info('Save model...')
                savepath = str(checkpoints_dir) + '/best_model.pth'
                log_string('Saving at %s' % savepath)
                state = {
                    'epoch': epoch,
                    'class_avg_iou': mIoU,
                    'model_state_dict': classifier.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(state, savepath)
                log_string('Saving model....')
            log_string('Best mIoU: %f' % best_iou)
        global_epoch += 1

    # At the end of training, log the best metrics
    log_string('<<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<<')
    log_string('Best Validation Results:')
    log_string(f'Val result: mIoU/mAcc/allAcc/mPrecision/mRecall {best_metrics["mIoU"]:.4f}/{best_metrics["mAcc"]:.4f}/{best_metrics["allAcc"]:.4f}/{best_metrics["mPrecision"]:.4f}/{best_metrics["mRecall"]:.4f}')
    
    # Check if class_ious, class_precision, and class_recall have valid data before accessing them
    if len(best_metrics["class_ious"]) > 0 and len(best_metrics["class_precision"]) > 0 and len(best_metrics["class_recall"]) > 0:
        for l in range(len(best_metrics["class_ious"])):  # Use len(best_metrics["class_ious"]) instead of NUM_CLASSES
            log_string(f'Class_{l} - {seg_label_to_cat[l]} Result: IoU={best_metrics["class_ious"][l]:.4f}, Precision={best_metrics["class_precision"][l]:.4f}, Recall={best_metrics["class_recall"][l]:.4f}')
    else:
        log_string("Warning: No valid class IoU, Precision, or Recall data available. Skipping per-class logging.")
    log_string('Saving final model...')
    savepath = str(checkpoints_dir) + '/final_model.pth'
    torch.save({
        'epoch': args.epoch,
        'model_state_dict': classifier.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, savepath)

    log_string('<<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<<')

if __name__ == '__main__':
    args = parse_args()
    main(args)
