
"""
Train MSFANet with the paper's settings (Sec. 4.1) using Torchreid (paper used Torchreid, PyTorch 1.7).
 
    pip install torch torchvision
    pip install git+https://github.com/KaiyangZhou/deep-person-reid.git   # or: pip install torchreid
 
    python train.py --dataset market1501 --root ./reid-data --mode scratch
    python train.py --dataset dukemtmcreid --root ./reid-data --mode scratch --blocks 222
    python train.py --dataset market1501 --root ./reid-data --mode pretrained --weights imagenet_msfanet.pth
"""
import argparse
import torch
import torchvision.transforms as T
import torchreid
from torchreid.data.transforms import Random2DTranslation, RandomPatch
 
from torchreid.models import msfanet as M
 
p = argparse.ArgumentParser()
p.add_argument('--dataset', default='market1501')        # market1501 | dukemtmcreid
p.add_argument('--root', default='reid-data')
p.add_argument('--mode', choices=['scratch', 'pretrained'], default='scratch')
p.add_argument('--blocks', choices=['123', '222'], default='123')
p.add_argument('--weights', default='')                  # your own ImageNet-pretrained weights
p.add_argument('--save_dir', default='log/msfanet')
p.add_argument('--eval_only', action='store_true')
args = p.parse_args()
 
H, W = 256, 128
mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
 
# Data augmentation (Sec. 3.2 / Table 2 last row):
# flip + crop + erase + patch + AutoAugment
train_tf = T.Compose([
    Random2DTranslation(H, W, p=0.5),                  # random crop (cut edges)
    T.RandomHorizontalFlip(),                          # flip
    T.AutoAugment(T.AutoAugmentPolicy.IMAGENET),       # autoaugment (policy not stated in paper)
    RandomPatch(),                                     # random patch (occlusion simulation)
    T.ToTensor(),
    T.Normalize(mean, std),
    T.RandomErasing(p=0.5, value=0),                   # random erase
])
test_tf = T.Compose([T.Resize((H, W)), T.ToTensor(), T.Normalize(mean, std)])
 
datamanager = torchreid.data.ImageDataManager(
    root=args.root, sources=args.dataset, targets=args.dataset,
    height=H, width=W,
    batch_size_train=128, batch_size_test=100,
    train_sampler='RandomIdentitySampler',             # P x K sampling needed for TriHard
    num_instances=4,
    workers=8,
)
datamanager.train_loader.dataset.transform = train_tf  # swap in our augmentation pipeline
datamanager.test_loader[args.dataset]['query'].dataset.transform = test_tf
datamanager.test_loader[args.dataset]['gallery'].dataset.transform = test_tf
 
build = M.msfanet_123 if args.blocks == '123' else M.msfanet_222
model = build(num_classes=datamanager.num_train_pids, loss='triplet')
if args.weights:
    sd = torch.load(args.weights, map_location='cpu')
    sd = {k: v for k, v in sd.items() if not k.startswith('classifier')}
    print(model.load_state_dict(sd, strict=False))
model = model.cuda()
 
n_params, flops = torchreid.utils.compute_model_complexity(model, (1, 3, H, W))
print(f'Params: {n_params/1e6:.2f}M  FLOPs: {flops/1e9:.2f}G  (paper: 2.52M / 0.82G for 1-2-3)')
 
if args.mode == 'scratch':
    optimizer = torchreid.optim.build_optimizer(
        model, optim='amsgrad', lr=0.065, weight_decay=5e-4)
    scheduler = torchreid.optim.build_lr_scheduler(
        optimizer, lr_scheduler='multi_step', stepsize=[150, 225, 300, 350], gamma=0.1)
    fixbase, open_layers = 0, None
else:
    optimizer = torchreid.optim.build_optimizer(
        model, optim='amsgrad', lr=0.0015, weight_decay=5e-4)
    scheduler = torchreid.optim.build_lr_scheduler(
        optimizer, lr_scheduler='cosine', max_epoch=400)
    fixbase, open_layers = 10, ['fc', 'classifier']    # first 10 epochs: train classifier only
 
# Fusion loss  L = 1.0 * softmax + 0.5 * TriHard  (Eq. 6, alpha=1, beta=0.5, Table 3)
engine = torchreid.engine.ImageTripletEngine(
    datamanager, model, optimizer, margin=0.3,
    weight_t=0.5, weight_x=1.0, scheduler=scheduler, label_smooth=False)
 
engine.run(
    save_dir=args.save_dir, max_epoch=400, eval_freq=10, print_freq=20,
    test_only=args.eval_only, fixbase_epoch=fixbase, open_layers=open_layers,
    start_eval=300,
)
 
