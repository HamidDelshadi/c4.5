import numpy as np
import collections
import logging
import turtle
import math

def log(msg):
    logging.info(msg)
class HDNode:
    def __init__(self, attribute=None, node_type=None, pred=None):
        self.type = node_type #leaf or inner
        self.attribute = attribute # only for inner
        self.pred = pred # only for leafs
        self.children = [] # list of children
        self.children_condition = [] # list of conditions for each child
        self.is_continuous = False
        self.count = 0

class C4d5:
    def __init__(self):
        log('tree is initialized')
        self.root = None
        self.threshold = 0.95
        self.max_depth = 7
        self.min_data_size = 1

    def __rec_draw(self, node,portion, height):
        turtle.begin_fill()
        turtle.circle(20)
        turtle.end_fill()
        turtle.write(str(node.attribute) if node.type=='inner' else str(node.pred))
        if len(node.children)==0:
            return
        child_portion =portion / len(node.children)
        
        offset = -portion/2 + child_portion/2
        pos = turtle.pos()
        for child in node.children:
            turtle.setpos(pos[0] + offset, pos[1] - height)
            self.__rec_draw(child,child_portion,height)
            turtle.up()
            turtle.setpos(pos)
            turtle.down()
            offset+=child_portion

    
    def draw(self):
        turtle.speed(0)
        turtle.tracer(0, 0)
        turtle.color('black','yellow')
        turtle.up()
        turtle.goto(0, 250)
        turtle.down()
        node = self.root
        turtle.shape('turtle')
        self.__rec_draw(node,1000,80)
        turtle.hideturtle()
        turtle.update()

    def calc_attr_h(self,x_attr, attr_values, attr_counts, y, y_values, w):
        splits = []
        p_relative =[]
        h_attr = 0
        split = 0
        x_known_count = x_attr.size
        if '?' in attr_values:
            x_known_count -= attr_counts[attr_values=='?'][0]
        f = x_known_count/x_attr.size
        w_splits = []
        missing_split = None
        missing_flag = False
        for value in attr_values:
            if value=='?':
                missing_split = np.argwhere(x_attr == value).flatten()
                missing_flag =True    
                temp = np.argwhere(x_attr == value)
                sp = temp.reshape(temp.size)
                attr_count = np.sum(w[sp])
            else:
                temp = np.argwhere(x_attr == value)
                sp = temp.reshape(temp.size)
                splits.append(sp)
                w_splits.append(w[sp])
                attr_count = np.sum(w[sp])
                h_attr_values = 0
                for y_val in y_values:
                    mask = np.zeros(w.size, np.bool)
                    mask[sp]=1
                    mask[y.squeeze()!=y_val]=0
                    p_temp = np.sum(w[mask])/attr_count
                    if p_temp !=0:
                        h_attr_values -= p_temp * np.log2(p_temp)
                p = attr_count/x_known_count
                h_attr += p*h_attr_values
                p_relative.append(p)
            p_split = attr_count/y.size
            split -= p_split*np.log2(p_split)

        if missing_flag:
            for i in range(len(splits)):
                for j in range(missing_split.size):
                    splits[i]= np.append(splits[i],missing_split[j])
                    w_splits[i] = np.append(w_splits[i], w[missing_split[j]]*p_relative[i])
        return split,splits,w_splits, h_attr, f

    def calc_continuous_attr_h(self,x_attr, attr_values,attr_counts, y, y_values,w,h):
        splits = []
        p_relative = []
        w_splits = []
        gain_ratios = []
        best_gr = 0
        best_splits = []
        avgs = []
        attr_known_values = attr_values[attr_values!='?'].flatten()
        a = float(attr_known_values[0])
        b = 0
        for i in range(1, attr_known_values.size):
            b = float(attr_known_values[i])
            avgs.append((a+b)/2)
            a = b
        if attr_known_values.size == 1:
            avgs.append(attr_known_values[0])
        x_known_count = x_attr.size
        if '?' in attr_values:
            x_known_count -= attr_counts[attr_values=='?'][0]
        f = x_known_count/x_attr.size
        for value in avgs:
            splits=[]
            w_splits =[]
            h_attr = 0
            split = 0
            temp = np.argwhere((x_attr <= value) & (x_attr!='?'))
            first_branch = temp.reshape(temp.size)
            splits.append(first_branch)
            w_splits.append(w[first_branch])
            temp = np.argwhere((x_attr > value)  & (x_attr!='?'))
            second_branch = temp.reshape(temp.size)
            branches = 2
            if (second_branch.size == 0):
                branches = 1
            else:
                splits.append(second_branch)
                w_splits.append(w[second_branch])
            for branch in range(branches):
                h_attr_values = 0
                attr_count = np.sum(w[splits[branch]])
                for y_val in y_values:
                    mask = np.zeros(w.size, np.bool)
                    mask[splits[branch]] =1
                    mask[y.squeeze()!=y_val] = 0
                    p_temp = np.sum(w[mask])/attr_count
                    if p_temp !=0:
                        h_attr_values -= p_temp * np.log2(p_temp)
                p = attr_count/x_known_count
                h_attr += p*h_attr_values
                p_relative.append(p)
                p_split = attr_count/y.size
                split -= p_split*np.log2(p_split)
            if '?' in attr_values:
                p_split = np.sum(w[x_attr == '?'])/y.size
                split -= p_split*np.log(p_split)
                missing_split = np.argwhere(x_attr == '?').squeeze()
                for i in range(len(splits)):
                    for j in range(missing_split.size):
                        splits[i] = np.append(splits[i],missing_split[j])
                        w_splits[i] = np.append(w_splits[i],w[missing_split[j]]*p_relative[i])
            gain = h - h_attr
            gain_ratio = gain/split
            if split == 0:
                gain_ratio = 0
            gain_ratios.append(gain_ratio)
            if gain_ratios[best_gr] <= gain_ratio:
                break_point = value
                best_splits = splits
                best_gr = len(gain_ratios)-1
                best_split = split
                best_h_attr = h_attr
                best_w_split = w_splits
        return best_split,best_splits,best_w_split, best_h_attr, f, break_point
        
    def __gain_ratio(self,x, y, w, attrs,continuous_attrs):

        n = x.shape[0]
        unkown_rows = np.unique(np.where(x == '?')[0])
        mask = np.ones(n, np.bool)
        mask[unkown_rows] = 0
        known_count = n - unkown_rows.size
        y_values = np.unique(y) #check if it is necessery
        y_known = y[mask]
        (y_known_values, y_known_counts) = np.unique(y_known,return_counts=True)
        #only counts the known values in calculating info(i)
        p = y_known_counts/known_count
        h = -np.sum((p*np.log2(p)))
        gain_ratios = []
        best_gr = 0
        best_split = []
        best_values = []
        continuous = False
        for attr_id, attr in enumerate(attrs):
            x_attr = x[:,attr]
            (attr_values, attr_counts) = np.unique(x_attr, return_counts=True)
            if attr in continuous_attrs:
                split,splits,w_split, h_attr,f, break_point = self.calc_continuous_attr_h(x_attr, attr_values,attr_counts, y, y_values,w,h)
            else:
                split,splits,w_split, h_attr,f = self.calc_attr_h(x_attr, attr_values,attr_counts, y, y_values,w)
            gain = f*(h - h_attr)
            gain_ratio = gain/split
            if split == 0:
                gain_ratio = 0
            gain_ratios.append(gain_ratio)
            if gain_ratios[best_gr] <= gain_ratio:
                best_split = splits
                best_gr = attr_id
                w_best_split = w_split
                if attr in continuous_attrs:
                    best_values = break_point
                    continuous = True
                else:
                    best_values = attr_values[attr_values!='?']
                    continuous = False
        return best_split, w_best_split, best_values, best_gr, continuous
               

    def __rec_train(self,node, x, y, w, attrs,continuous_attrs, depth):
        splits, w_split, conditions, max_id, is_continuous = self.__gain_ratio(x, y, w, attrs, continuous_attrs)
        node.type = "inner"
        node.is_continuous = is_continuous
        node.attribute = attrs[max_id]
        node.children_condition = conditions
        remaining_attr = np.delete(attrs, max_id) if not is_continuous else np.copy(attrs)
        for idx, split in enumerate(splits):
            child = HDNode()
            node.children.append(child)
            y_split = y[split]
            values = np.unique(y_split)
            counts = np.array([np.sum(w_split[idx][y_split.squeeze()==value]) for value in values])
            max_count = np.max(counts)
            if len(values)==1 or len(remaining_attr)==0 or max_count/split.size > self.threshold or depth>=self.max_depth or split.size<self.min_data_size:
                child.type = "leaf"
                child.pred = values[np.argmax(counts)]
                child.count = max_count
                continue
            self.__rec_train(child,x[split],y_split,w_split[idx], remaining_attr,continuous_attrs, depth+1)
            
    def train(self, x_tr, y_tr, params):
        continuous_attrs = self.continuous_attrs
        self.threshold = params[0]
        self.max_depth = params[1]
        self.min_data_size = params[2]
        n, d = x_tr.shape
        log("function_start train, n={},d={}".format(n,d))
        self.root = HDNode()
        node = self.root
        w = np.ones(n)
        self.__rec_train(node, x_tr, y_tr,w, np.arange(d), continuous_attrs,1)
            
    def __traversal(self, x):
        node = self.root
        while True:
            if node.type == "inner":
                if node.is_continuous:
                    if x[node.attribute] <= node.children_condition or len(node.children)==1: 
                    #it's better if you remove the or ###########
                        node = node.children[0]
                    else:
                        node = node.children[1]
                else:
                    node = node.children[np.where(node.children_condition ==x[node.attribute])[0][0]]
            else:
                return node.pred

    def __traversal_rec(self, x, node):
        if node.type == "inner":
            if (x[node.attribute] == '?') or ((not node.is_continuous) and (x[node.attribute] not in node.children_condition)):
                preds = np.array([])
                counts = np.array([])
                for child in node.children:
                    pred, count = self.__traversal_rec(x, child)
                    if preds.size > 0:
                        preds = np.concatenate((preds, pred))
                        counts = np.concatenate((counts, count))
                    else:
                        preds = pred
                        counts = count
                return preds, counts
            if node.is_continuous:
                if x[node.attribute] <= node.children_condition or len(node.children)==1: 
                #it's better if you remove the or ###########
                    return self.__traversal_rec(x,node.children[0])
                else:
                    return self.__traversal_rec(x,node.children[1])
            else:
                return self.__traversal_rec(x,node.children[np.where(node.children_condition.flatten() ==x[node.attribute])[0][0]])
        else:
            return np.array([node.pred]), np.array([node.count])

    def predict_single(self, x):
        preds, pred_counts = self.__traversal_rec(x, self.root)
        values= np.unique(preds)
        best = 0
        best_count = 0
        for value in values:
            count = np.sum(pred_counts[preds == value])
            if count >= best_count:
                best_count= count
                best = value
        return best

    def predict(self, x_set):
        return np.array([self.predict_single(x) for x in x_set])