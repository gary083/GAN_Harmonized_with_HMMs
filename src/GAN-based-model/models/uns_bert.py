from .base import ModelBase
from evalution import frame_eval


class UnsBertModel(ModelBase):

    def train(
            self,
            config,
            data_loader,
            dev_data_loader=None,
            aug=False,
    ):
        print('TRAINING(unsupervised)...')
        if aug:
            get_target_batch = data_loader.get_aug_target_batch
        else:
            get_target_batch = data_loader.get_target_batch

        batch_size = config.batch_size * config.repeat
        step_gen_loss, step_dis_loss, step_seg_loss = 0.0, 0.0, 0.0
        max_fer = 100.0
        frame_temp = 0.9
        for step in range(1, config.step + 1):
            batch_sample_feat, batch_sample_len, batch_repeat_num = data_loader.get_sample_batch(
                config.batch_size,
                repeat=config.repeat,
            )
            batch_target_idx, batch_target_len = get_target_batch(batch_size)

            if step % config.print_step == 0:
                print(
                    f'Step: {step:5d} dis_loss: {step_gen_loss:.4f} gen_loss: {step_dis_loss:.4f} seg_loss: {step_seg_loss:.4f}')
                step_gen_loss, step_dis_loss, step_seg_loss = 0.0, 0.0, 0.0

            if step % config.eval_step == 0:
                # step_fer = frame_eval(self.sess, self, dev_data_loader)
                # print(f'EVAL max: {max_fer:.2f} step: {step_fer:.2f}')
                # if step_fer < max_fer:
                #     max_fer = step_fer
                # TODO
                pass

        print('=' * 80)

    def restore(self, save_dir):
        pass

    def output_framewise_prob(self, output_path, data_loader):
        pass
