import numpy as np 
import pandas as pd 
# IMPORTANT: DO NOT USE ANY OTHER 3RD PARTY PACKAGES
# (math, random, collections, functools, etc. are perfectly fine)


class DecisionTree:
    
    def __init__(self):
        # NOTE: Feel free add any hyperparameters 
        # (with defaults) as you see fit
        pass
    
    def fit(self, train_data):
        
        # Tree to be built
        tree = {}
        
        # These are the yes and nos.
        class_list = train_data['Play Tennis'].unique()
        
        # Generate will run recursively untill the tree is fitted.
        generate_tree(tree, None, train_data, class_list)

        print("FITTING COMPLETED, tree")
        
    def predict(self, X):
        """
        Generates predictions
        
        Note: should be called after .fit()
        
        Args:
            X (pd.DataFrame): an mxn discrete matrix where
                each row is a sample and the columns correspond
                to the features.
            
        Returns:
            A length m vector with predictions
        """
        # TODO: Implement 
        raise NotImplementedError()
    
    def get_rules(self):
        """
        Returns the decision tree as a list of rules
        
        Each rule is given as an implication "x => y" where
        the antecedent is given by a conjuction of attribute
        values and the consequent is the predicted label
        
            attr1=val1 ^ attr2=val2 ^ ... => label
        
        Example output:
        >>> model.get_rules()
        [
            ([('Outlook', 'Overcast')], 'Yes'),
            ([('Outlook', 'Rain'), ('Wind', 'Strong')], 'No'),
            ...
        ]
        """
        # TODO: Implement
        raise NotImplementedError()


# --- Some utility functions 
    
def accuracy(y_true, y_pred):
    """
    Computes discrete classification accuracy
    
    Args:
        y_true (array<m>): a length m vector of ground truth labels
        y_pred (array<m>): a length m vector of predicted labels
        
    Returns:
        The average number of correct predictions
    """
    assert y_true.shape == y_pred.shape
    return (y_true == y_pred).mean()


def entropy(counts):
    """
    Computes the entropy of a partitioning
    
    Args:
        counts (array<k>): a lenth k int array >= 0. For instance,
            an array [3, 4, 1] implies that you have a total of 8
            datapoints where 3 are in the first group, 4 in the second,
            and 1 one in the last. This will result in entropy > 0.
            In contrast, a perfect partitioning like [8, 0, 0] will
            result in a (minimal) entropy of 0.0
            
    Returns:
        A positive float scalar corresponding to the (log2) entropy
        of the partitioning.
    
    """
    assert (counts >= 0).all()
    probs = counts / counts.sum()
    probs = probs[probs > 0]  # Avoid log(0)
    return - np.sum(probs * np.log2(probs))


def count_all_feats(X, y):
    X.reset_index(inplace=True, drop=True)
    y.reset_index(inplace=True, drop=True)
    
    # Initialize dictionaries to count "Yes" and "No" for each feature
    feat_counts = {
        "Outlook": {"Yes": [0, 0, 0], "No": [0, 0, 0]},
        "Temperature": {"Yes": [0, 0, 0], "No": [0, 0, 0]},
        "Humidity": {"Yes": [0, 0], "No": [0, 0]},
        "Wind": {"Yes": [0, 0], "No": [0, 0]}
    }
    
    # Iterate through the data to count each features attributes count
    for i in range(X.shape[0]):
        outlook, temperature, humidity, wind = X.loc[i]
        # Determine the target ("Yes" or "No")
        target = y[i]
        # Increment the corresponding counts
        feat_counts["Outlook"][target][["Sunny", "Rain", "Overcast"].index(outlook)] += 1
        feat_counts["Temperature"][target][["Hot", "Mild", "Cool"].index(temperature)] += 1
        feat_counts["Humidity"][target][["High", "Normal"].index(humidity)] += 1
        feat_counts["Wind"][target][["Strong", "Weak"].index(wind)] += 1
        
    return feat_counts

def get_max_info_gain_feat(feat_counts, total_days, source_entropy):
    max_info_gain = -1 
    max_info_feat = None

    # Calculate entropies
    for feat in feat_counts:
        feat_entropy = 0
        number_of_attributes_for_feat = len(feat_counts[feat]["Yes"])
        for attribute_number in range(0, number_of_attributes_for_feat):
            yes = feat_counts[feat]["Yes"][attribute_number]
            no = feat_counts[feat]["No"][attribute_number]
            counts = np.array([ yes, no ])
            attribute_entropy = entropy(counts) 
            total_counts_for_attribute = yes+no
            feat_entropy += (total_counts_for_attribute/total_days)*attribute_entropy

        # Find reduction in randomness a.k.a. information gain
        feat_info_gain = source_entropy - feat_entropy
        if(feat_info_gain > max_info_gain):
            max_info_gain = feat_info_gain
            max_info_feat = feat    
    return max_info_feat

def get_source_entropy(y):
    yes = 0 
    no = 0
    if 'No' in y.unique():
        no = y.value_counts()['No']
    if 'Yes' in y.unique():
        yes = y.value_counts()['Yes']
    return entropy(np.array([yes, no]))

def gen_sub_tree(max_info_gain_feat, train_data, class_list):
    # Inspired by: https://medium.com/geekculture/step-by-step-decision-tree-id3-algorithm-from-scratch-in-python-no-fancy-library-4822bbfdd88f
    
    tree = {} 

    # Dictionary of the count of unique feature value
    attribute_count_dict = train_data[max_info_gain_feat].value_counts(sort=False)
    a_number = 0

    for attribute, attribute_total_count in attribute_count_dict.iteritems():
        assigned_to_node = False #flag for tracking feature_value is pure class or not
        attribute_data = train_data[train_data[max_info_gain_feat] == attribute]

        for c in class_list: #for each class
            class_count = 0
            try:
                class_count = attribute_data[train_data["Play Tennis"] == c].shape[0] #count of class c
            except:
                print("Pure class has been found") # (for the opposite class)
            if class_count == attribute_total_count: # Pure class = contains only one class
                tree[attribute] = c #adding node to the tree
                train_data = train_data[train_data[max_info_gain_feat] != attribute] #removing rows with feature_value
                #train_data.reset_index()
                assigned_to_node = True
        if not assigned_to_node: #not pure class
            tree[attribute] = "?" #as feature_value is not a pure class, it should be expanded further, 
                                      #so the branch is marking with ?
    return tree, train_data

def generate_tree(root, prev_attribute, train_data, class_list):
    if train_data.shape[0] != 0:
        # Find the feature counts
        feat_counts = count_all_feats(train_data.drop(columns=['Play Tennis']), train_data['Play Tennis'])

        # Get total number of days
        datapoints = len(train_data.index) 

        # Calculate entropy at source
        source_entropy = get_source_entropy(train_data["Play Tennis"])

        # Find the max info gain
        max_info_gain_feat = get_max_info_gain_feat(feat_counts, datapoints, source_entropy)

        # Get sub tree and updated train data where columns with pure classes are removed.
        tree, train_data = gen_sub_tree(max_info_gain_feat, train_data, class_list)

        next_root = None

        if prev_attribute != None:
            root[prev_attribute] = dict()
            root[prev_attribute][max_info_gain_feat] = tree
            next_root = root[prev_attribute][max_info_gain_feat]
        else:
            root[max_info_gain_feat] = tree
            next_root = tree

        for node, branch in list(next_root.items()):
            print("node",node)
            if branch == "?":
                attribute_data = train_data[train_data[max_info_gain_feat] == node]
                generate_tree(next_root, node, attribute_data, class_list)
    else:
        return None