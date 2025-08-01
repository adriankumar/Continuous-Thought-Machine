import numpy as np

#generates single arithmetic progression sample with random start and step
def generate_arithmetic_sample(n=5):
    start = np.random.randint(0, 51)  #range 0 to 50
    step = np.random.randint(-10, 11)  #range -10 to +10
    while step == 0:  #exclude step size 0
        step = np.random.randint(-10, 11)
    
    #generate sequence of length 5
    sequence = [start + i * step for i in range(n)]
    
    #input is first 4 numbers, label is 5th number
    input_seq = sequence[:4]
    label = sequence[4]
    
    return input_seq, label

#creates complete dataset with specified number of samples
def create_arithmetic_dataset(num_samples, sequence_length=5):
    inputs = []
    labels = []
    
    for _ in range(num_samples):
        input_seq, label = generate_arithmetic_sample(sequence_length)
        inputs.append(input_seq)
        labels.append(label)
    
    return np.array(inputs), np.array(labels)

#generate or load arithmetic sequence dataset
def prepare_dataset(config, sequence_length=5):
 
    train_data, train_labels = create_arithmetic_dataset(config['train_samples'], sequence_length)
    test_data, test_labels = create_arithmetic_dataset(config['test_samples'], sequence_length)
         
    print(f"generated new dataset: {len(train_data)} train, {len(test_data)} test samples")
    
    return train_data, train_labels, test_data, test_labels


def show_dataset_examples(inputs, labels, num_examples=5):
    print(f"\nfirst {num_examples} training examples:")
    for i in range(min(num_examples, len(inputs))):
        sequence = inputs[i]
        target = labels[i]
        step = sequence[1] - sequence[0]  #calculate step size
        print(f"  {sequence} → {target} (step: {step})")