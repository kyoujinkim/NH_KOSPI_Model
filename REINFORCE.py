import numpy as np
import pickle
import tensorflow as tf
import random
from tqdm import tqdm
import os
import os.path
import datetime
from sklearn.preprocessing import StandardScaler
import multiprocessing

scaler = StandardScaler()

def preprocFeatures(arr):
    gainArr = arr.iloc[:, 0].copy()
    scaled_arr = scaler.transform(arr)
    return gainArr, scaled_arr


def get_variables(train_size, direct):
    with open(direct, 'rb') as f:
        arr = pickle.load(f)
    arr = arr[:train_size]
    return arr.values.tolist()


def discount_rewards(r, random_rate):
    discounted_r = np.zeros_like(r)
    for t in range(0, r.size):
        discounted_r[t] = r[t] / r.std()
    for t in range(0, r.size):
        discounted_r[t] = random.choices([discounted_r[t], -discounted_r[t]],
                                         weights=[1000 - random_rate, random_rate],
                                         k=1)
        # give false information to escape from false local optimal
    return discounted_r


def step(a, arr, arr_idx, fwd_idx):
    gain = arr[arr_idx + fwd_idx] - arr[arr_idx]
    if a == 2:  # Bull
        r = gain
    elif a == 1:  # Neutral
        r = -gain / 100
    else:  # Bear
        r = -gain
    return r


def reinforceLearning(train_date):

    model_code = train_date[10:]
    train_date = train_date[:10]
    save_direct = basedir + '/weights/reinforce_v2'
    save_direct_weight = save_direct + '/weight' + model_code + str(train_date) + '.h5'
    save_direct_log = save_direct + '/log' + model_code + str(train_date) + '.txt'

    if os.path.isfile(save_direct_weight):
        return

    inputdata_direct = basedir+ '/pickle_var/variables/' + model_code + str(train_date) + '.pkl'
    with open(inputdata_direct, 'rb') as f:
        variables = pickle.load(f)

    scaler.fit(variables)
    gainArr, variables = preprocFeatures(variables)
    variables = variables.tolist()
    gainArr = gainArr.values.tolist()

    former_mean = 0
    random_rate = 1
    inputdim = hist * len(variables[-1])

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(252, input_dim=inputdim, activation='relu'))
    model.add(tf.keras.layers.Dense(63, activation='relu'))
    model.add(tf.keras.layers.Dense(3, activation='softmax'))
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    compute_loss = tf.keras.losses.SparseCategoricalCrossentropy()

    gradBuffer = model.trainable_variables
    for ix, grad in enumerate(gradBuffer):
        gradBuffer[ix] = grad * 0

    scores = []
    iter_log = []
    with tqdm(range(iterations)) as tqd:
        for iter in tqd:

            memory = []
            score = 0
            if learning_range == 'all':
                startpoint = hist + random.randrange(0, int(hist/2))
            else:
                startpoint = max(len(variables) - learning_range + random.randrange(0, int(hist/2)),
                                 hist + random.randrange(0, int(hist/2)))

            fwd_idx = 5

            for idx in range(startpoint, len(variables) - fwd_idx, fwd_idx):
                s = tf.expand_dims(np.concatenate(variables[idx - hist:idx], axis=None), 0)
                if idx + fwd_idx >= len(variables):
                    break
                with tf.GradientTape() as tape:
                    logits = model(s)
                    a_dist = logits.numpy()
                    a = np.random.choice(a_dist[0], p=a_dist[0])
                    a = np.argmax(a_dist == a)
                    loss = compute_loss([a], logits)
                grads = tape.gradient(loss, model.trainable_variables)
                r = step(a, gainArr, idx, fwd_idx)
                score += r
                memory.append([grads, r])

            scores.append(score)
            memory = np.array(memory)
            memory[:, 1] = discount_rewards(memory[:, 1], random_rate)

            for grads, r in memory:
                for ix, grad in enumerate(grads):
                    gradBuffer[ix] += grad * r

            if iter % update_period == 0:

                EarlyStopped = False
                if iter > 200 and np.std(scores[-100:]) >= np.std(scores[-200:-100]) and np.mean(scores[-100:]) <= np.mean(
                        scores[-200:-100]):
                    EarlyStopped = True
                    break

                tqd.set_postfix(Time=train_date, Score=np.mean(scores[-100:]), STD=np.std(scores[-100:]),
                                MAX=np.max(scores[-100:]), Min=np.min(scores[-100:]), MC = model_code, ES=EarlyStopped)
                iter_log.append("{} Learning  {}  Score  {}   Var  {}   Max  {}   Min  {}"
                                .format(train_date, iter, np.mean(scores[-100:]), np.std(scores[-100:]),
                                        np.max(scores[-100:]), np.min(scores[-100:])))
                if former_mean == np.mean(scores[-50:]):
                    random_rate = 100
                else:
                    random_rate = 1
                former_mean = np.mean(scores[-50:])

            if iter % update_period == 0:
                optimizer.apply_gradients(zip(gradBuffer, model.trainable_variables))
                for ix, grad in enumerate(gradBuffer):
                    gradBuffer[ix] = grad * 0

    model.save_weights(save_direct_weight)
    with open(save_direct_log, 'w') as f:
        for line in iter_log:
            f.write(str(line) + '\n')

#tuning basic model parameters
basedir = 'C:/PycharmProjects/pythonProject/textAnalysis'
hist = 126
model_code_array = ['excEcon', 'excFdmt', 'excSent', 'onlySent','onlyEcon','onlyFdmt']
#['All', 'excEcon', 'excFdmt', 'excSent', 'onlySent','onlyEcon','onlyFdmt']
iterations = 10001
update_period = 50
learning_range = 'all'

if __name__ == '__main__':

    model_load = False

    list = os.listdir(basedir + '/pickle_var/variables')

    for model_code in model_code_array:
        model_list = [i for i in list if i[:len(model_code)] == model_code]
        new_list = []
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        new_list.append(model_list[0].replace(model_code, '').replace('.pkl', ''))
        new_list.append(model_list[0].replace(model_code, '').replace('.pkl', ''))
        for el in model_list:
            new_el = el.replace(model_code, '').replace('.pkl', '')
            if int(new_el[-2:]) >= int(new_list[-1][-2:]):
                new_list[-1] = new_el
            else:
                new_list.append(new_el)

        BaseDate = datetime.datetime.strptime('2012-03-16', '%Y-%m-%d')

        new_list = [i + model_code for i in new_list]
        pool = multiprocessing.Pool(os.cpu_count())
        pool.map(reinforceLearning, new_list)
