import numpy as np
import random as rd
import scipy.sparse as sp
from time import time

class Data(object):
    def __init__(self, path, batch_size):
        self.path = path
        self.batch_size = batch_size

        train_file = path + '/train.txt'
        test_file = path + '/test.txt'
        MP_emb_file_genre = path + '/author.txt'
        MP_emb_file_country = path + '/publisher.txt'
        MP_emb_file_director = path + '/user.txt'
        MP_emb_file_actor = path + '/year.txt'

        #get number of users and items
        self.n_users, self.n_items = 0, 0
        self.n_train, self.n_test = 0, 0
        self.neg_pools = {}
        self.exist_users = []

        with open(train_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                        l = l.strip('\n').split(' ')
                        items = [int(i) for i in l[1:]]
                        uid = int(l[0])
                        if uid < 12771:
                            self.exist_users.append(uid)
                        #self.exist_users=self.exist_users[:12642]
                        self.n_items = max(self.n_items, max(items))
                        self.n_users = max(self.n_users, uid)
                        self.n_train += len(items)

        with open(test_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    #if len(l)<100:
                        l = l.strip('\n')
                        try:
                            items = [int(i) for i in l.split(' ')[1:]]
                        except Exception:
                            continue
                        self.n_items = max(self.n_items, max(items))
                        self.n_test += len(items)
        self.n_items += 1
        self.n_users += 1

        self.print_statistics()

        self.R = sp.dok_matrix((self.n_users, self.n_items), dtype=np.float32)

        self.train_items, self.test_set = {}, {}
        with open(train_file) as f_train:
            with open(test_file) as f_test:
                for l in f_train.readlines():
                    if len(l) == 0: break
                    l = l.strip('\n')
                    items = [int(i) for i in l.split(' ')]
                    uid, train_items = items[0], items[1:]

                    if uid < 12771:
                        for i in train_items:
                            self.R[uid, i] = 1.

                        self.train_items[uid] = train_items

                for l in f_test.readlines():
                    if len(l) == 0: break
                    l = l.strip('\n')
                    try:
                        items = [int(i) for i in l.split(' ')]
                    except Exception:
                        continue

                    uid, test_items = items[0], items[1:]
                    if uid < 12771:
                        self.test_set[uid] = test_items


        A1_list_temp = []
        A2_list_temp = []
        A3_list_temp = []
        A4_list_temp = []
        with open (MP_emb_file_genre) as f_MP_vec:
            emb_temp=[]
            for l in f_MP_vec.readlines()[2:]:
                if len(l) > 0:
                        l = l.strip('\n').split(' ',1)
                        A1_list_temp.append(l)
                        emb=l[1].strip("\n").strip('\r').split(' ')[:-1]
                        emb_temp.append([float(x) for x in emb])
        A1_dic_vec_id =list(dict(A1_list_temp).keys())
        A1_dic_vec=dict(zip(A1_dic_vec_id,emb_temp))

        with open (MP_emb_file_country) as f_MP_vec:
            emb_temp=[]
            for l in f_MP_vec.readlines()[2:]:
                if len(l) > 0:
                        l = l.strip('\n').split(' ',1)
                        A2_list_temp.append(l)
                        emb=l[1].strip("\n").strip('\r').split(' ')[:-1]
                        emb_temp.append([float(x) for x in emb])
        A2_dic_vec_id =list(dict(A2_list_temp).keys())
        A2_dic_vec=dict(zip(A2_dic_vec_id,emb_temp))

        
        with open (MP_emb_file_director) as f_MP_vec:
            emb_temp=[]
            for l in f_MP_vec.readlines()[2:]:
                if len(l) > 0:
                        l = l.strip('\n').split(' ',1)
                        A3_list_temp.append(l)
                        emb=l[1].strip("\n").strip('\r').split(' ')[:-1]
                        emb_temp.append([float(x) for x in emb])
        A3_dic_vec_id =list(dict(A3_list_temp).keys())
        A3_dic_vec=dict(zip(A3_dic_vec_id,emb_temp))
        
        
        with open (MP_emb_file_actor) as f_MP_vec:
            emb_temp=[]
            for l in f_MP_vec.readlines()[2:]:
                if len(l) > 0:
                        l = l.strip('\n').split(' ',1)
                        A4_list_temp.append(l)
                        emb=l[1].strip("\n").strip('\r').split(' ')[:-1]
                        emb_temp.append([float(x) for x in emb])
        A4_dic_vec_id =list(dict(A4_list_temp).keys())
        A4_dic_vec=dict(zip(A4_dic_vec_id,emb_temp))
        

        A1_vec_uid=[]
        A1_vec_aspect_id=[]
        A2_vec_uid=[]
        A2_vec_aspect_id=[]
        A3_vec_uid=[]
        A3_vec_aspect_id=[]
        A4_vec_uid=[]
        A4_vec_aspect_id=[]

        for i in A1_dic_vec_id:
            if i[0] == 'a':
                A1_vec_uid.append(int(i[1:]))
            else:
                A1_vec_aspect_id.append(str(i[1:]))

        print('you are using aspect from:',MP_emb_file_genre)

        for i in A2_dic_vec_id:
            if i[0] == 'a':
                A2_vec_uid.append(int(i[1:]))
            else:
                A2_vec_aspect_id.append(str(i[1:])) #0 if no identifier of "a" or "v"

        print('you are using aspect from:',MP_emb_file_country)

        for i in A3_dic_vec_id:
            if i[0] == 'a':
                A3_vec_uid.append(int(i[1:]))
            else:
                A3_vec_aspect_id.append(str(i[1:]))

        print('you are using aspect from:',MP_emb_file_director)

        
        for i in A4_dic_vec_id:
            if i[0] == 'a':
                A4_vec_uid.append(int(i[1:]))
            else:
                A4_vec_aspect_id.append(str(i[1:]))

        print('you are using aspect from:',MP_emb_file_actor)
        


        A1_vec_user=[]
        A1_vec_aspect=[]
        for key in A1_dic_vec.keys():
            if key[0]=='a':
            
                A1_vec_user.append(A1_dic_vec[key])
            else:
                
                A1_vec_aspect.append(A1_dic_vec[key])
        A1_vec_user=np.array(A1_vec_user,dtype='float32')
        A1_vec_aspect=np.array(A1_vec_aspect,dtype='float32')


        A2_vec_user=[]
        A2_vec_aspect=[]
        for key in A2_dic_vec.keys():
            if key[0]=='a':
               
                A2_vec_user.append(A2_dic_vec[key])
            else:
               
                A2_vec_aspect.append(A2_dic_vec[key])
        A2_vec_user=np.array(A2_vec_user,dtype='float32')
        A2_vec_aspect=np.array(A2_vec_aspect,dtype='float32')

        A3_vec_user=[]
        A3_vec_aspect=[]
        for key in A3_dic_vec.keys():
            if key[0]=='a':
                
                A3_vec_user.append(A3_dic_vec[key])
            else:
                
                A3_vec_aspect.append(A3_dic_vec[key])
        A3_vec_user=np.array(A3_vec_user,dtype='float32')
        A3_vec_aspect=np.array(A3_vec_aspect,dtype='float32')

        
        A4_vec_user=[]
        A4_vec_aspect=[]
        for key in A4_dic_vec.keys():
            if key[0]=='a':
               
                A4_vec_user.append(A4_dic_vec[key])
            else:
               
                A4_vec_aspect.append(A4_dic_vec[key])
        A4_vec_user=np.array(A4_vec_user,dtype='float32')
        A4_vec_aspect=np.array(A4_vec_aspect,dtype='float32')
        

        self.A1_vec_uid_num = len(A1_vec_uid)
        self.A1_vec_aspect_id_num = len(A1_vec_aspect_id)
        self.A1_vec_uid = A1_vec_uid
        self.A1_vec_aspect_id = A1_vec_aspect_id

        self.A2_vec_uid_num = len(A2_vec_uid)
        self.A2_vec_aspect_id_num = len(A2_vec_aspect_id)
        self.A2_vec_uid = A2_vec_uid
        self.A2_vec_aspect_id = A2_vec_aspect_id

        self.A3_vec_uid_num = len(A3_vec_uid)
        self.A3_vec_aspect_id_num = len(A3_vec_aspect_id)
        self.A3_vec_uid = A3_vec_uid
        self.A3_vec_aspect_id = A3_vec_aspect_id

        
        self.A4_vec_uid_num = len(A4_vec_uid)
        self.A4_vec_aspect_id_num = len(A4_vec_aspect_id)
        self.A4_vec_uid = A4_vec_uid
        self.A4_vec_aspect_id = A4_vec_aspect_id
        

        self.A1_vec=A1_dic_vec
        self.A2_vec=A2_dic_vec
        self.A3_vec=A3_dic_vec
        self.A4_vec=A4_dic_vec


        self.A1_vec_user=A1_vec_user
        self.A1_vec_aspect=A1_vec_aspect
        self.A2_vec_user=A2_vec_user
        self.A2_vec_aspect=A2_vec_aspect     
        self.A3_vec_user=A3_vec_user
        self.A3_vec_aspect=A3_vec_aspect
        self.A4_vec_user=A4_vec_user
        self.A4_vec_aspect=A4_vec_aspect
                
    def get_adj_mat(self):
        try:
            t1 = time()
            adj_mat = sp.load_npz(self.path + '/s_adj_mat.npz')
            norm_adj_mat = sp.load_npz(self.path + '/s_norm_adj_mat.npz')
            mean_adj_mat = sp.load_npz(self.path + '/s_mean_adj_mat.npz')
            print('already load adj matrix', adj_mat.shape, time() - t1)

        except Exception:
            adj_mat, norm_adj_mat, mean_adj_mat = self.create_adj_mat()
            sp.save_npz(self.path + '/s_adj_mat.npz', adj_mat)
            sp.save_npz(self.path + '/s_norm_adj_mat.npz', norm_adj_mat)
            sp.save_npz(self.path + '/s_mean_adj_mat.npz', mean_adj_mat)
            
        try:
            pre_adj_mat = sp.load_npz(self.path + '/s_pre_adj_mat.npz')
        except Exception:
            adj_mat=adj_mat
            rowsum = np.array(adj_mat.sum(1))
            d_inv = np.power(rowsum, -0.5).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)

            norm_adj = d_mat_inv.dot(adj_mat)
            norm_adj = norm_adj.dot(d_mat_inv)
            print('generate pre adjacency matrix.')
            pre_adj_mat = norm_adj.tocsr()
            sp.save_npz(self.path + '/s_pre_adj_mat.npz', norm_adj)


        return adj_mat, norm_adj_mat, mean_adj_mat, pre_adj_mat

    def create_adj_mat(self):
        t1 = time()
        adj_mat = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)
        adj_mat = adj_mat.tolil()
        R = self.R.tolil()

        adj_mat[:self.n_users, self.n_users:] = R
        adj_mat[self.n_users:, :self.n_users] = R.T
        adj_mat = adj_mat.todok()
        print('already create adjacency matrix', adj_mat.shape, time() - t1)

        t2 = time()

        def normalized_adj_single(adj):
            rowsum = np.array(adj.sum(1))

            d_inv = np.power(rowsum, -1).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)

            norm_adj = d_mat_inv.dot(adj)
            # norm_adj = adj.dot(d_mat_inv)
            print('generate single-normalized adjacency matrix.')
            return norm_adj.tocoo()

        def check_adj_if_equal(adj):
            dense_A = np.array(adj.todense())
            degree = np.sum(dense_A, axis=1, keepdims=False)

            temp = np.dot(np.diag(np.power(degree, -1)), dense_A)
            print('check normalized adjacency matrix whether equal to this laplacian matrix.')
            return temp

        norm_adj_mat = normalized_adj_single(adj_mat + sp.eye(adj_mat.shape[0]))
        mean_adj_mat = normalized_adj_single(adj_mat)

        print('already normalize adjacency matrix', time() - t2)
        return adj_mat.tocsr(), norm_adj_mat.tocsr(), mean_adj_mat.tocsr()

    def negative_pool(self):
        t1 = time()
        for u in self.train_items.keys():
            neg_items = list(set(range(self.n_items)) - set(self.train_items[u]))
            pools = [rd.choice(neg_items) for _ in range(100)]
            self.neg_pools[u] = pools
        print('refresh negative pools', time() - t1)

    def sample(self):
        if self.batch_size <= self.n_users:
            users = rd.sample(self.exist_users, self.batch_size)
        else:
            users = [rd.choice(self.exist_users) for _ in range(self.batch_size)]


        def sample_pos_items_for_u(u, num):
            pos_items = self.train_items[u]
            n_pos_items = len(pos_items)
            pos_batch = []
            while True:
                if len(pos_batch) == num: break
                pos_id = np.random.randint(low=0, high=n_pos_items, size=1)[0]
                pos_i_id = pos_items[pos_id]

                if pos_i_id not in pos_batch:
                    pos_batch.append(pos_i_id)
            return pos_batch

        def sample_neg_items_for_u(u, num):
            neg_items = []
            while True:
                if len(neg_items) == num: break
                neg_id = np.random.randint(low=0, high=self.n_items,size=1)[0]
                if neg_id not in self.train_items[u] and neg_id not in neg_items:
                    neg_items.append(neg_id)
            return neg_items

        def sample_neg_items_for_u_from_pools(u, num):
            neg_items = list(set(self.neg_pools[u]) - set(self.train_items[u]))
            return rd.sample(neg_items, num)

        pos_items, neg_items = [], []
        for u in users:
            pos_items += sample_pos_items_for_u(u, 1)
            neg_items += sample_neg_items_for_u(u, 1)

        #print('from load_data:',users,len(users))

        return users, pos_items, neg_items
    
    def sample_test(self):
        if self.batch_size <= self.n_users:
            users = rd.sample(self.test_set.keys(), self.batch_size)
        else:
            users = [rd.choice(self.exist_users) for _ in range(self.batch_size)]

        def sample_pos_items_for_u(u, num):
            pos_items = self.test_set[u]
            n_pos_items = len(pos_items)
            pos_batch = []
            while True:
                if len(pos_batch) == num: break
                pos_id = np.random.randint(low=0, high=n_pos_items, size=1)[0]
                pos_i_id = pos_items[pos_id]

                if pos_i_id not in pos_batch:
                    pos_batch.append(pos_i_id)
            return pos_batch

        def sample_neg_items_for_u(u, num):
            neg_items = []
            while True:
                if len(neg_items) == num: break
                neg_id = np.random.randint(low=0, high=self.n_items, size=1)[0]
                if neg_id not in (self.test_set[u]+self.train_items[u]) and neg_id not in neg_items:
                    neg_items.append(neg_id)
            return neg_items
    
        def sample_neg_items_for_u_from_pools(u, num):
            neg_items = list(set(self.neg_pools[u]) - set(self.train_items[u]))
            return rd.sample(neg_items, num)

        pos_items, neg_items = [], []
        for u in users:
            pos_items += sample_pos_items_for_u(u, 1)
            neg_items += sample_neg_items_for_u(u, 1)

        return users, pos_items, neg_items  
    
    def get_num_users_items(self):
        return self.n_users, self.n_items

    def print_statistics(self):
        print('n_users=%d, n_items=%d' % (self.n_users, self.n_items))
        print('n_interactions=%d' % (self.n_train + self.n_test))
        print('n_train=%d, n_test=%d, sparsity=%.5f' % (self.n_train, self.n_test, (self.n_train + self.n_test)/(self.n_users * self.n_items)))

    def get_sparsity_split(self):
        try:
            split_uids, split_state = [], []
            lines = open(self.path + '/sparsity.split', 'r').readlines()

            for idx, line in enumerate(lines):
                if idx % 2 == 0:
                    split_state.append(line.strip())
                    print(line.strip())
                else:
                    split_uids.append([int(uid) for uid in line.strip().split(' ')])
            print('get sparsity split.')

        except Exception:
            split_uids, split_state = self.create_sparsity_split()
            f = open(self.path + '/sparsity.split', 'w')
            for idx in range(len(split_state)):
                f.write(split_state[idx] + '\n')
                f.write(' '.join([str(uid) for uid in split_uids[idx]]) + '\n')
            print('create sparsity split.')

        return split_uids, split_state

    def create_sparsity_split(self):
        all_users_to_test = list(self.test_set.keys())
        user_n_iid = dict()

        # generate a dictionary to store (key=n_iids, value=a list of uid).
        for uid in all_users_to_test:
            train_iids = self.train_items[uid]
            test_iids = self.test_set[uid]

            n_iids = len(train_iids) + len(test_iids)

            if n_iids not in user_n_iid.keys():
                user_n_iid[n_iids] = [uid]
            else:
                user_n_iid[n_iids].append(uid)
        split_uids = list()

        # split the whole user set into four subset.
        temp = []
        count = 1
        fold = 4
        n_count = (self.n_train + self.n_test)
        n_rates = 0

        split_state = []
        for idx, n_iids in enumerate(sorted(user_n_iid)):
            temp += user_n_iid[n_iids]
            n_rates += n_iids * len(user_n_iid[n_iids])
            n_count -= n_iids * len(user_n_iid[n_iids])

            if n_rates >= count * 0.25 * (self.n_train + self.n_test):
                split_uids.append(temp)

                state = '#inter per user<=[%d], #users=[%d], #all rates=[%d]' %(n_iids, len(temp), n_rates)
                split_state.append(state)
                print(state)

                temp = []
                n_rates = 0
                fold -= 1

            if idx == len(user_n_iid.keys()) - 1 or n_count == 0:
                split_uids.append(temp)

                state = '#inter per user<=[%d], #users=[%d], #all rates=[%d]' % (n_iids, len(temp), n_rates)
                split_state.append(state)
                print(state)



        return split_uids, split_state


