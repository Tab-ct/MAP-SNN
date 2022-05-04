import torch
import torch.nn as nn
import time
import matplotlib.pyplot as plt




def init_weights(m):
    if type(m) == nn.Linear:
        print(f" [=> Linear Layer Init <=]")
        torch.nn.init.xavier_normal_(m.weight)

def print_results(acc_record, loss_test_record):
    plt.plot(range(len(acc_record)), acc_record)
    plt.xlabel('epoch')
    plt.ylabel('acc[%]')
    plt.show()
    plt.plot(range(len(loss_test_record)), loss_test_record)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.show()


def print_model_params(snn):
    # print('+++++++++++ Params Checking Example: +++++++++++')
    # print(f'+++++++++++    h2-h_thresh: {snn.h2.h_thresh}')
    # print(f'+++++++++++    h2-decay: {snn.h2.h_decay_sigmoid(snn.h2.h_decay)}')
    # print(f'+++++++++++    h2-decay: {snn.h2.h_decay}')
    # print(f'+++++++++++    h2-decay_const: {snn.h2.decay_const}')
    # print(f'+++++++++++    h2-inh: {snn.h2.h_inh}')
    # print(f'+++++++++++    conv1-a: {snn.conv1.kernel_parameter_a[0:5]}')
    # print(f'+++++++++++    conv1-b: {snn.conv1.kernel_parameter_b[0:5]}')
    # print(f'+++++++++++    conv1-tau: {snn.conv1.kernel_parameter_tau[0:5]}')
    # print(f'+++++++++++    conv1-decay: {snn.conv1.kernel_time_shift[0:5]}')
    # print(f'+++++++++++    out_w.sum(0): {snn.out_w.sum(0)}')
    # print(f'+++++++++++    out_w[0]: {snn.out_w[0]}')
    # print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    return 


def test_snn(snn, args, test_loader, device=None):
    correct = 0
    total = 0
    loss_total = 0
    if args.criterion == "MSE":
        criterion = nn.MSELoss()
    else:
        criterion =  nn.CrossEntropyLoss(reduction='mean')
    with torch.no_grad():
            for step, (test_data,test_label) in enumerate(test_loader):
                test_label = test_label.to(device)
                test_data = test_data.float().to(device)
                outputs = snn(test_data)
                _, predicted = outputs.max(1)

                batch_size = test_data.shape[0]
                if args.criterion == "MSE":
                    test_label = test_label.long() 
                    labels_ = torch.zeros(batch_size, args.n_classes).to(device).scatter_(1, test_label.view(-1, 1), 1)
                    loss = criterion(outputs, labels_.float())
                else:
                    loss = criterion(outputs, test_label.reshape(-1).long())

                loss_total += loss.item()
                
                total += float(test_label.shape[0])
                correct += float(predicted.eq(test_label.squeeze(-1)).sum().item())

            acc = 100. * float(correct) / float(total)

            ##### SHOW PARAMs OF SNN #####
            print_model_params(snn)  #####
            ##############################

            
    return acc, loss_total




def train_snn(snn, args, train_loader, test_loader, device=None):

    acc_record = list([])
    loss_test_record = list([])

    snn = snn.apply(init_weights)
    if args.criterion == "MSE":
        criterion = nn.MSELoss()
        print(f"Using MSE as criterion")
    else:
        criterion = nn.CrossEntropyLoss(reduction='mean')
        print(f"Using CrossEntropy as criterion")
    
    if args.optimizer == "Adam":
        optimizer = torch.optim.Adam(snn.parameters(), lr=args.lr)
        print(f"Using Adam as optimizer")
    if args.optimizer == "SGD":
        optimizer = torch.optim.SGD(snn.parameters(), lr=args.lr, momentum=0.9)
        print(f"Using SGD as optimizer")
    step_count = 0
    steps_per_epoch = len(train_loader)
    acc_record.append(0)

    for epoch in range(args.num_epochs):
        running_loss = 0
        start_time = time.time()
        total = 0
        correct = 0

        for step, (input_data,labels) in enumerate(train_loader):

            optimizer.zero_grad()
            batch_size = input_data.shape[0]

            input_data = input_data.float().to(device)
            labels = labels.to(device)

            
            outputs = snn(input_data)

            _, predicted = outputs.max(1)
            total += float(labels.shape[0])
            correct += float(predicted.eq(labels.squeeze(-1)).sum().item())

            if args.criterion == "MSE":
                labels = labels.long()
                labels_ = torch.zeros(batch_size, args.n_classes).to(device).scatter_(1, labels.view(-1, 1), 1)
                loss = criterion(outputs, labels_.float())
            else:
                loss = criterion(outputs, labels.reshape(-1).long())
            running_loss += loss.item()
            loss.backward()
            step_count += 1
            optimizer.step()


            # PRINT_BLOCK########################################
            if (step+1) % int(steps_per_epoch/args.train_acc_print_time) == 0:
                print ('\nEpoch [%d/%d], Step [%d/%d], Loss: %.5f'
                        %(epoch+1, args.num_epochs, step+1, len(train_loader),running_loss ))
                running_loss = 0
                print('Accuracy:', correct/total )
                print('Time elasped:', time.time()-start_time)
                correct = 0
                total = 0
            # PRINT_BLOCK########################################
        
        print('Iters:', epoch)
        test_acc, test_loss = test_snn(snn, args, test_loader, device=device)
        acc_record.append(test_acc)
        loss_test_record.append(test_loss)
        print('Test Accuracy on test dataset: %.3f' % (test_acc))  
        print('Test Loss on test dataset: %.3f' % (test_loss))  
        print('Time elasped:', time.time()-start_time)
        print(f"Maximum accuracy on test dataset: {max(acc_record)}")
        print('\n\n\n')
        print('\n\n\n')

    return acc_record, loss_test_record