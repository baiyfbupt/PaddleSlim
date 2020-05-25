from itertools import izip
import paddle.fluid as fluid
from paddleslim.teachers.bert.reader.cls import *
from paddleslim.nas.darts.search_space import AdaBERTClassifier

import logging
from paddleslim.common import AvgrageMeter, get_logger
logger = get_logger(__name__, level=logging.INFO)


def main():
    place = fluid.CUDAPlace(0)

    BERT_BASE_PATH = "./data/pretrained_models/uncased_L-12_H-768_A-12"
    bert_config_path = BERT_BASE_PATH + "/bert_config.json"
    vocab_path = BERT_BASE_PATH + "/vocab.txt"
    data_dir = "./data/glue_data/MNLI/"
    teacher_model_dir="./teacher_model/steps_23000.pdparams"
    num_imgs = 392702
    max_seq_len = 512
    do_lower_case = True
    batch_size = 32
    epoch = 30

    processor = MnliProcessor(
        data_dir=data_dir,
        vocab_path=vocab_path,
        max_seq_len=max_seq_len,
        do_lower_case=do_lower_case,
        in_tokens=False)

    train_reader = processor.data_generator(
        batch_size=batch_size,
        phase='train',
        epoch=epoch,
        dev_count=1,
        shuffle=True)

    val_reader = processor.data_generator(
        batch_size=batch_size,
        phase='train',
        epoch=epoch,
        dev_count=1,
        shuffle=True)



    with fluid.dygraph.guard(place):
        model = AdaBERTClassifier(
            8,
            teacher_model=teacher_model_dir,
            data_dir=data_dir
        )

        def train_one_epoch(model, train_loader, valid_loader, optimizer, epoch):
            ce_losses = AvgrageMeter()
            accs = AvgrageMeter()
            #e_losses = AvgrageMeter()
            model.train()

            step_id = 0
            for train_data in train_loader():
                loss, acc = model.loss(train_data)
                loss.backward()

                optimizer.minimize(loss)
                model.clear_gradients()

                batch_size = train_data[0].shape[0]
                ce_losses.update(loss.numpy(), batch_size)
                accs.update(acc.numpy(), batch_size)
                # e_losses.update(e_loss.numpy(), batch_size)

                if step_id % 10 == 0:
                    logger.info("Train Epoch {}, Step {}, Lr {:.6f} loss {:.6f}; acc: {:.6f};".format(epoch, step_id, optimizer.current_step_lr(), ce_losses.avg[0], accs.avg[0]))
                step_id += 1

        model_parameters = [
            p for p in model.parameters()
            if p.name not in [a.name for a in model.arch_parameters()]
        ]
        step_per_epoch = int(num_imgs / batch_size)
        learning_rate = fluid.dygraph.CosineDecay(
            0.02, step_per_epoch, epoch)
        optimizer = fluid.optimizer.MomentumOptimizer(
            learning_rate,
            0.9,
            regularization=fluid.regularizer.L2DecayRegularizer(3e-4),
            parameter_list=model_parameters)

        train_loader = fluid.io.DataLoader.from_generator(
            capacity=64,
            use_double_buffer=True,
            iterable=True,
            return_list=True)
        valid_loader = fluid.io.DataLoader.from_generator(
            capacity=64,
            use_double_buffer=True,
            iterable=True,
            return_list=True)

        train_loader.set_batch_generator(train_reader, places=place)
        valid_loader.set_batch_generator(val_reader, places=place)

        for epoch_id in range(epoch):
            logger.info('Epoch {}, lr {:.6f}'.format(
                epoch_id, optimizer.current_step_lr()))

            train_one_epoch(model, train_loader, valid_loader, optimizer, epoch_id)



if __name__ == '__main__':
    main()
