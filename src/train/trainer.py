from torch.cuda.amp import autocast, GradScaler
import os
from tqdm import tqdm
import torch
from src.loss import loss as loss_module
import torch.optim as optimizer_module
import torch.optim.lr_scheduler as scheduler_module


METRIC_NAMES = {
    'RMSELoss': 'RMSE',
    'MSELoss': 'MSE',
    'MAELoss': 'MAE'
}

def train(args, model, dataloader, logger, setting):
    
    if args.wandb:
        import wandb
    
    scaler = GradScaler()   # AMP를 위한 GradScaler 초기화
    minimum_loss = None
    loss_fn = getattr(loss_module, args.loss)().to(args.device)
    args.metrics = sorted([metric for metric in set(args.metrics) if metric != args.loss])

    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = getattr(optimizer_module, args.optimizer.type)(trainable_params, **args.optimizer.args)

    if args.lr_scheduler.use:
        args.lr_scheduler.args = {k: v for k, v in args.lr_scheduler.args.items() 
                                  if k in getattr(scheduler_module, args.lr_scheduler.type).__init__.__code__.co_varnames}
        lr_scheduler = getattr(scheduler_module, args.lr_scheduler.type)(optimizer, **args.lr_scheduler.args)
    else:
        lr_scheduler = None

    for epoch in range(args.train.epochs):
        model.train()
        total_loss, train_len = 0, len(dataloader['train_dataloader'])

        # tqdm을 사용하여 학습 진행 상황 표시
        progress_bar = tqdm(dataloader['train_dataloader'], desc=f'[Epoch {epoch+1}/{args.train.epochs}]', unit="batch")

        for batch_idx, data in enumerate(progress_bar):
            # 멀티모달 데이터 처리
            if args.model_args[args.model].datatype == 'image':
                x, y = [data['user_book_vector'].to(args.device), data['img_vector'].to(args.device)], data['rating'].to(args.device)
            elif args.model_args[args.model].datatype == 'text':
                x, y = [data['user_book_vector'].to(args.device), data['user_summary_vector'].to(args.device), data['book_summary_vector'].to(args.device)], data['rating'].to(args.device)
            elif args.model_args[args.model].datatype == 'multimodal':  # 멀티모달 모델 처리
                x, y = [data['images'].to(args.device), data['input_ids'].to(args.device), data['attention_mask'].to(args.device), data['numerical_features'].to(args.device)], data['rating'].to(args.device)
            else:
                x, y = data[0].to(args.device), data[1].to(args.device)

            optimizer.zero_grad()
            
            # Automatic Mixed Precision 사용
            with autocast():
                y_hat = model(x)
                loss = loss_fn(y_hat, y.float())

            # GradScaler로 backward 및 optimizer 업데이트
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

            # tqdm의 진행 상태 업데이트
            progress_bar.set_postfix({'Batch Loss': loss.item()})

        if args.lr_scheduler.use and args.lr_scheduler.type != 'ReduceLROnPlateau':
            lr_scheduler.step()
        
        # 에폭 별 손실 값 출력
        train_loss = total_loss / train_len
        print(f'Epoch [{epoch+1}/{args.train.epochs}] - Train Loss: {train_loss:.4f}')

        if args.dataset.valid_ratio != 0:  # valid 데이터가 존재할 경우
            valid_loss = valid(args, model, dataloader['valid_dataloader'], loss_fn)
            print(f'Validation Loss: {valid_loss:.4f}')
            if args.lr_scheduler.use and args.lr_scheduler.type == 'ReduceLROnPlateau':
                lr_scheduler.step(valid_loss)

            # 기타 메트릭 출력 (RMSE, MSE, MAE 등)
            valid_metrics = dict()
            for metric in args.metrics:
                metric_fn = getattr(loss_module, metric)().to(args.device)
                valid_metric = valid(args, model, dataloader['valid_dataloader'], metric_fn)
                valid_metrics[f'Valid {METRIC_NAMES[metric]}'] = valid_metric

            for metric, value in valid_metrics.items():
                print(f'{metric}: {value:.4f}')

            logger.log(epoch=epoch+1, train_loss=train_loss, valid_loss=valid_loss, valid_metrics=valid_metrics)

            if args.wandb:
                wandb.log({f'Train {METRIC_NAMES[args.loss]}': train_loss, 
                           f'Valid {METRIC_NAMES[args.loss]}': valid_loss, **valid_metrics})
        else:
            logger.log(epoch=epoch+1, train_loss=train_loss)
            if args.wandb:
                wandb.log({f'Train {METRIC_NAMES[args.loss]}': train_loss})
        
        # 가장 좋은 모델 저장
        if args.train.save_best_model:
            best_loss = valid_loss if args.dataset.valid_ratio != 0 else train_loss
            if minimum_loss is None or minimum_loss > best_loss:
                minimum_loss = best_loss
                os.makedirs(args.train.ckpt_dir, exist_ok=True)
                torch.save(model.state_dict(), f'{args.train.ckpt_dir}/{setting.save_time}_{args.model}_best.pt')
        else:
            os.makedirs(args.train.ckpt_dir, exist_ok=True)
            torch.save(model.state_dict(), f'{args.train.ckpt_dir}/{setting.save_time}_{args.model}_e{epoch:02}.pt')

    logger.close()

    return model


def valid(args, model, dataloader, loss_fn):
    model.eval()
    total_loss = 0

    for data in dataloader:
        if args.model_args[args.model].datatype == 'image':
            x, y = [data['user_book_vector'].to(args.device), data['img_vector'].to(args.device)], data['rating'].to(args.device)
        elif args.model_args[args.model].datatype == 'text':
            x, y = [data['user_book_vector'].to(args.device), data['user_summary_vector'].to(args.device), data['book_summary_vector'].to(args.device)], data['rating'].to(args.device)
        elif args.model_args[args.model].datatype == 'multimodal':
            x, y = [data['images'].to(args.device), data['input_ids'].to(args.device), data['attention_mask'].to(args.device), data['numerical_features'].to(args.device)], data['rating'].to(args.device)
        else:
            x, y = data[0].to(args.device), data[1].to(args.device)

        # AMP 수정 반영
        with autocast():
            y_hat = model(x)
            loss = loss_fn(y.float(), y_hat)
            total_loss += loss.item()

    return total_loss / len(dataloader)



def test(args, model, dataloader, setting, checkpoint=None):
    predicts = list()

    if checkpoint:
        model.load_state_dict(torch.load(checkpoint, weights_only=True))
    else:
        if args.train.save_best_model:
            model_path = f'{args.train.ckpt_dir}/{setting.save_time}_{args.model}_best.pt'
        else:
            model_path = f'{args.train.save_dir.checkpoint}/{setting.save_time}_{args.model}_e{args.train.epochs-1:02d}.pt'
        model.load_state_dict(torch.load(model_path, weights_only=True))
    
    model.eval()
    
    with torch.no_grad():  # test에서는 gradient 계산이 필요 없으므로 no_grad 사용
        for data in dataloader['test_dataloader']:
            if args.model == 'MultiModalModel':
                # 멀티모달 데이터 처리
                image, input_ids, attention_mask, numerical_features = (
                    data['images'].to(args.device), 
                    data['input_ids'].to(args.device), 
                    data['attention_mask'].to(args.device), 
                    data['numerical_features'].to(args.device)
                )
                with autocast():  # AMP 적용
                    y_hat = model(image, input_ids, attention_mask, numerical_features)
            else:
                if args.model_args[args.model].datatype == 'image':
                    x = [data['user_book_vector'].to(args.device), data['img_vector'].to(args.device)]
                elif args.model_args[args.model].datatype == 'text':
                    x = [data['user_book_vector'].to(args.device), data['user_summary_vector'].to(args.device), data['book_summary_vector'].to(args.device)]
                else:
                    x = data[0].to(args.device)
                
                with autocast():  # AMP 적용
                    y_hat = model(x)
                    
            predicts.extend(y_hat.tolist())

    return predicts
