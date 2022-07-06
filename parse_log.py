import os
import numpy as np

log_dir = 'workdir'
model_names = ['DAN']
train_names = [[0]]
group_names = [[1, 2, 3, 4]]
test_names = [[1, 2, 3, 4, 5]]
class_names = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]]

F_scores_dict = np.zeros(
    (len(model_names), len(train_names[0]), len(group_names[0]),
     len(test_names[0]), len(class_names[0])))
finetune_F_scores = np.zeros(
    (len(model_names), len(train_names[0]), len(group_names[0]),
     len(test_names[0]), len(class_names[0])))
J_scores_dict = np.zeros(
    (len(model_names), len(train_names[0]), len(group_names[0]),
     len(test_names[0]), len(class_names[0])))
finetune_J_scores = np.zeros(
    (len(model_names), len(train_names[0]), len(group_names[0]),
     len(test_names[0]), len(class_names[0])))


def parse_test_log(log_file):
    is_F_score_exist = None
    is_J_score_exist = None
    F_scores = []
    J_scores = []
    F_mean = 0
    J_mean = 0

    if not os.path.exists(log_file):
        print('{} does not exist'.format(log_file))
    with open(log_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            # if line start with 'F:'
            if line.startswith('F:'):
                F_scores = line.split('\n')[0].split(' ')[1:]
                F_mean = float(F_scores[0])
                F_scores = [float(F_score) for F_score in F_scores[2:]]
                is_F_score_exist = True
            elif line.startswith('J:'):
                J_scores = line.split('\n')[0].split(' ')[1:]
                J_mean = float(J_scores[0])
                J_scores = [float(J_score) for J_score in J_scores[2:]]
                is_J_score_exist = True
    return is_F_score_exist, F_scores, F_mean, is_J_score_exist, J_scores, J_mean


for model_id, model_name in enumerate(model_names):
    for train_id, train_name in enumerate(train_names[model_id]):
        for group_id, group_name in enumerate(group_names[model_id]):
            for test_id, test_name in enumerate(test_names[model_id]):
                log_file = os.path.join(
                    log_dir, model_name,
                    'id_{}_group_{}_of_{}'.format(train_name, group_name,
                                                  len(group_names[model_id])),
                    'test_best_{}.txt'.format(test_name))

                if not os.path.exists(log_file):
                    print(
                        '[{},train{},group{},test{}] {} does not exist'.format(
                            model_name, train_name, group_name, test_name,
                            log_file))
                    continue

                F_flag, F_scores, F_mean, J_flag, J_scores, J_mean = parse_test_log(
                    log_file)

                if not F_flag:
                    print('[{},train{},group{},test{}] cannot find F score'.
                          format(model_name, train_name, group_name,
                                 test_name))
                    continue
                elif not J_flag:
                    print('[{},train{},group{},test{}] cannot find J score'.
                          format(model_name, train_name, group_name,
                                 test_name))
                    continue
                else:
                    if len(F_scores) != 10:
                        print(
                            '[{},train{},group{},test{}] F score length is not 10'
                            .format(model_name, train_name, group_name,
                                    test_name))
                    F_scores_dict[model_id, train_id, group_id,
                                  test_id, :] = F_scores
                    J_scores_dict[model_id, train_id, group_id,
                                  test_id, :] = J_scores

                for class_id, class_name in enumerate(class_names[model_id]):
                    log_file = os.path.join(
                        log_dir, model_name, 'id_{}_group_{}_of_{}'.format(
                            train_name, group_name,
                            len(group_names[model_id])),
                        '{}'.format(class_name), 'test_{}'.format(test_name),
                        'finetune_{}_test_{}.txt'.format(
                            class_name, test_name))
                    if not os.path.exists(log_file):
                        print(
                            '[{},train{},group{},test{},class{}] {} does not exist'
                            .format(model_name, train_name, group_name,
                                    test_name, class_name, log_file))
                        continue

                    F_flag, F_scores, F_mean, J_flag, J_scores, J_mean = parse_test_log(
                        log_file)
                    if not F_flag:
                        print(
                            '[{},train{},group{},test{}] cannot find F score'.
                            format(model_name, train_name, group_name,
                                   test_name))
                        continue
                    elif not J_flag:
                        print(
                            '[{},train{},group{},test{}] cannot find J score'.
                            format(model_name, train_name, group_name,
                                   test_name))
                        continue
                    else:
                        if len(F_scores) != 1:
                            print(
                                '[{},train{},group{},test{},class{}] F score length is not 10'
                                .format(model_name, train_id, group_id,
                                        test_id, class_id))
                        finetune_F_scores[model_id, train_id, group_id,
                                          test_id, class_id] = F_scores[0]
                        finetune_J_scores[model_id, train_id, group_id,
                                          test_id, class_id] = J_scores[0]

print('F_scores_dict: ', F_scores_dict)
print('finetune_F_scores: ', finetune_F_scores)

# save the results
result_file = os.path.join('result.csv')
with open(result_file, 'w') as f:
    f.write(
        'model,train,group,test,class,F_score,J_score,finetune_F_score,finetune_J_score,\n'
    )
    for model_id, model_name in enumerate(model_names):
        for train_id, train_name in enumerate(train_names[model_id]):
            for group_id, group_name in enumerate(group_names[model_id]):
                for test_id, test_name in enumerate(test_names[model_id]):
                    for class_id, class_name in enumerate(
                            class_names[model_id]):
                        f.write('{}, {}, {}, {}, {}, {}, {}, {}, {},\n'.format(
                            model_name, train_name, group_name, test_name,
                            class_name, F_scores_dict[model_id, train_id,
                                                      group_id, test_id,
                                                      class_id],
                            J_scores_dict[model_id, train_id, group_id,
                                          test_id, class_id],
                            finetune_F_scores[model_id, train_id, group_id,
                                              test_id, class_id],
                            finetune_J_scores[model_id, train_id, group_id,
                                              test_id, class_id]))
                        f.flush()

    f.close()
