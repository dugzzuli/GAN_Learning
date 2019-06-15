import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

'''
第一步
设置超参数
'''

tf.set_random_seed(1)
np.random.seed(1)

## 设置超参数
BATCH_SIZE = 64		# 批量大小
LR_G = 0.0001           # 生成器的学习率
LR_D = 0.0001           # 判别器的学习率
N_IDEAS = 5             # 认为这是生成5种艺术作品（5种初始化曲线）
ART_COMPONENTS = 15     # 在画上画15个点练成一条线
PAINT_POINTS = np.vstack([np.linspace(-1, 1, ART_COMPONENTS) for _ in range(BATCH_SIZE)])
#列表解析式代替了for循环，PAINT_POINTS.shape=(64,15),
#np.vstack()默认逐行叠加（axis=0）

'''
第二步
专家开始作画
'''
def artist_works():    
    a = np.random.uniform(1, 2, size=BATCH_SIZE)[:, np.newaxis]
    #a为64个1到2均匀分布抽取的值，shape=(64,1)
    paintings = a * np.power(PAINT_POINTS, 2) + (a-1)
    return paintings

'''
第三步
设置生成器网络结构，生成一副业余画家的画
'''
with tf.variable_scope('Generator_Scope'):
    G_in = tf.placeholder(tf.float32, [None, N_IDEAS]) # 随机的ideals（来源于正态分布）
    G_l1 = tf.layers.dense(G_in, 128, tf.nn.relu)
    G_out = tf.layers.dense(G_l1, ART_COMPONENTS)      # 生成一副业余专家的画（15个数据点）

'''
第四步
设置判别器网络结构，先输入专家画，返回判断真的概率，再输入业余专家的画，同样返回判为真概率
'''

with tf.variable_scope('Discriminator_Scope'):
    """判别器与生成器不同，生成器只需要输入生成的数据就行，它无法接触到专家的画，
    如果能输入专家的画，那就不用学习了，直接导入到判别器就是0.5的概率，换句话说，
    生成器只能通过生成器的误差反馈来调节权重，使得逐渐生成逼真的画出来。"""
	
    # 接受专家的画
    real_art = tf.placeholder(tf.float32, [None, ART_COMPONENTS], name='real_in')   
	
    # 将专家的画输入到判别器，判别器判断这副画来自于专家的概率
    D_l0 = tf.layers.dense(real_art, 128, tf.nn.relu, name='Discri')
    prob_artist0 = tf.layers.dense(D_l0, 1, tf.nn.sigmoid, name='out')  
    
    # 之后输入业余专家的画，G_out代入到判别器中。
    D_l1 = tf.layers.dense(G_out, 128, tf.nn.relu, name='Discri', reuse=True)  
	
    # 代入生成的画，判别器判断这副画来自于专家的概率
    prob_artist1 = tf.layers.dense(D_l1, 1, tf.nn.sigmoid, name='out', reuse=True) 
    """注意到，判别器中当输入业余专家的画时，这层是可以重复利用的，通过动态调整这次的权重来完成判别器的loss最小，关键一步。"""

'''
第五步
定义误差loss
'''
#判别器loss，此时需同时优化两部分的概率
D_loss = -tf.reduce_mean(tf.log(prob_artist0) + tf.log(1-prob_artist1))

#对于生成器的loss，此时prob_artist0是固定的，可以看到生成器并没有输入专家的画，
#所以tf.log(prob_artist0)是一个常数，故在这里不用考虑。
G_loss = tf.reduce_mean(tf.log(1-prob_artist1))

'''
第六步
定义Train_D和Train_G
'''
train_D = tf.train.AdamOptimizer(LR_D).minimize(
    D_loss, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Discriminator_Scope'))
train_G = tf.train.AdamOptimizer(LR_G).minimize(
    G_loss, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Generator_Scope'))

'''
第七步
定义sess，初始化所以变量
'''
sess = tf.Session()
sess.run(tf.global_variables_initializer())

'''
第八步
画图，实时展示结果
'''
plt.ion()   # 连续画图
for step in range(5000):
    artist_paintings = artist_works()           # 专家的画，每一轮专家的画都是随机生成的！
    G_ideas = np.random.randn(BATCH_SIZE, N_IDEAS)  #业余画家的5个想法
    G_paintings, pa0, Dl = sess.run([G_out, prob_artist0, D_loss, train_D, train_G],
     {G_in: G_ideas, real_art: artist_paintings})[:3]   # 训练和获取结果
                                    
    if step % 500 == 0:  # 每50步训练画一次图
        plt.cla()
        plt.plot(PAINT_POINTS[0], G_paintings[0], c='#4AD631', lw=3, label='生成的画',)
        plt.plot(PAINT_POINTS[0], artist_paintings[0], c='#4AD632', lw=3, label='专家的画',)
        plt.plot(PAINT_POINTS[0], 2 * np.power(PAINT_POINTS[0], 2) + 1, c='#74BCFF', lw=3, label='上限')
        plt.plot(PAINT_POINTS[0], 1 * np.power(PAINT_POINTS[0], 2) + 0, c='#FF9359', lw=3, label='下限')
        plt.text(-.5, 2.3, 'D accuracy=%.2f (0.5 for D to converge)' % pa0.mean(), fontdict={'size': 15})
        plt.text(-.5, 2, 'D score= %.2f (-1.38 for G to converge)' % -Dl, fontdict={'size': 15})
        plt.ylim((0, 3)); plt.legend(loc='upper right', fontsize=12); plt.draw(); plt.pause(0.01)

plt.ioff()
plt.show()



