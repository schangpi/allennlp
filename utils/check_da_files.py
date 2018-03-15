import os
import numpy as np

def get_accuracy(model, domains):
    accuracy = []
    for dm in domains:
        filepath = '../results/da_' + model + '_' + dm
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                for line in f.readlines():
                    pass
            try:
                accuracy.append(float(line.split()[-1]))
            except:
                accuracy.append(0.0)
        else:
            accuracy.append(0.0)
    print(accuracy)
    return sum(accuracy) / len(accuracy)

def get_avg(l):
    return round(100*np.array(l).mean(), 2)

def get_std(l):
    return round(100*np.array(l).std(), 2)

mz_domains = ["xpos_mz-sinorama"]
bc_domains = ["xpos_bc-cctv", "xpos_bc-msnbc", "xpos_bc-p2.5-c2e", "xpos_bc-cnn", "xpos_bc-p2.5-a2e", "xpos_bc-phoenix"]
nw_domains = ["xpos_nw-p2.5-a2e", "xpos_nw-wsj", "xpos_nw-xinhua", "xpos_nw-p2.5-c2e"]
wb_domains = ["xpos_wb-a2e", "xpos_wb-eng", "xpos_wb-p2.5-c2e" "xpos_wb-c2e" "xpos_wb-p2.5-a2e" "xpos_wb-sel"]
bn_domains = ["xpos_bn-p2.5-a2e", "xpos_bn-abc", "xpos_bn-p2.5-c2e", "xpos_bn-cnn",
              "xpos_bn-pri", "xpos_bn-mnb", "xpos_bn-voa", "xpos_bn-nbc"]

single_models = ["xpos_uni_tagger_crf_xpos",
                 "xpos_uni_tagger_crf_xpos0",
                 "xpos_uni_tagger_crf_xpos1"]
multi_models = ["multitagger_multi_all",
                "multitagger_multi_all0",
                "multitagger_multi_all1"]
teonly_models = ["taskembtagger_taskonly_embedding_tagger_all",
                 "taskembtagger_taskonly_embedding_tagger_all0",
                 "taskembtagger_taskonly_embedding_tagger_all1"]
tpeonly_models = ["taskembtagger_taskonly_prepend_embedding_tagger_all",
                  "taskembtagger_taskonly_prepend_embedding_tagger_all0",
                  "taskembtagger_taskonly_prepend_embedding_tagger_all1"]

mz_accs = []
bc_accs = []
nw_accs = []
wb_accs = []
bn_accs = []
for model in single_models:
    print(model)
    mz_accs.append(get_accuracy(model, mz_domains))
    bc_accs.append(get_accuracy(model, bc_domains))
    nw_accs.append(get_accuracy(model, nw_domains))
    wb_accs.append(get_accuracy(model, wb_domains))
    bn_accs.append(get_accuracy(model, bn_domains))
print(get_avg(mz_accs))
print(get_avg(bc_accs))
print(get_avg(nw_accs))
print(get_avg(wb_accs))
print(get_avg(bn_accs))

latex_str = ''
latex_str += 'STL (\\task{xpos}) '
latex_str += '& ' + str(get_avg(nw_accs)) + ' \\tiny{ $\pm$ '+ str(get_std(nw_accs)) + ' }'
latex_str += '& ' + str(get_avg(bc_accs)) + ' \\tiny{ $\pm$ '+ str(get_std(bc_accs)) + ' }'
latex_str += '& ' + str(get_avg(bn_accs)) + ' \\tiny{ $\pm$ '+ str(get_std(bn_accs)) + ' }'
latex_str += '& ' + str(get_avg(mz_accs)) + ' \\tiny{ $\pm$ '+ str(get_std(mz_accs)) + ' }'
latex_str += '& ' + str(get_avg(wb_accs)) + ' \\tiny{ $\pm$ '+ str(get_std(wb_accs)) + ' }'
latex_str += '& ' + str(get_avg([get_avg(nw_accs),
                                 get_avg(bc_accs),
                                 get_avg(bn_accs),
                                 get_avg(mz_accs),
                                 get_avg(wb_accs)])/100)
latex_str += '\\\\ \\hline\n'

mz_accs = []
bc_accs = []
nw_accs = []
wb_accs = []
bn_accs = []
for model in multi_models:
    print(model)
    mz_accs.append(get_accuracy(model, mz_domains))
    bc_accs.append(get_accuracy(model, bc_domains))
    nw_accs.append(get_accuracy(model, nw_domains))
    wb_accs.append(get_accuracy(model, wb_domains))
    bn_accs.append(get_accuracy(model, bn_domains))
print(get_avg(mz_accs))
print(get_avg(bc_accs))
print(get_avg(nw_accs))
print(get_avg(wb_accs))
print(get_avg(bn_accs))
latex_str += '\\textbf{Multi-Dec} (\\task{all}) '
latex_str += '& ' + str(get_avg(nw_accs)) + ' \\tiny{ $\pm$ '+ str(get_std(nw_accs)) + ' }'
latex_str += '& ' + str(get_avg(bc_accs)) + ' \\tiny{ $\pm$ '+ str(get_std(bc_accs)) + ' }'
latex_str += '& ' + str(get_avg(bn_accs)) + ' \\tiny{ $\pm$ '+ str(get_std(bn_accs)) + ' }'
latex_str += '& ' + str(get_avg(mz_accs)) + ' \\tiny{ $\pm$ '+ str(get_std(mz_accs)) + ' }'
latex_str += '& ' + str(get_avg(wb_accs)) + ' \\tiny{ $\pm$ '+ str(get_std(wb_accs)) + ' }'
latex_str += '& ' + str(get_avg([get_avg(nw_accs),
                                 get_avg(bc_accs),
                                 get_avg(bn_accs),
                                 get_avg(mz_accs),
                                 get_avg(wb_accs)])/100)
latex_str += '\\\\ \\hline\n'

mz_accs = []
bc_accs = []
nw_accs = []
wb_accs = []
bn_accs = []
for model in teonly_models:
    print(model)
    mz_accs.append(get_accuracy(model, mz_domains))
    bc_accs.append(get_accuracy(model, bc_domains))
    nw_accs.append(get_accuracy(model, nw_domains))
    wb_accs.append(get_accuracy(model, wb_domains))
    bn_accs.append(get_accuracy(model, bn_domains))
print(get_avg(mz_accs))
print(get_avg(bc_accs))
print(get_avg(nw_accs))
print(get_avg(wb_accs))
print(get_avg(bn_accs))
latex_str += '\\textbf{TE-Dec} (\\task{all}) '
latex_str += '& ' + str(get_avg(nw_accs)) + ' \\tiny{ $\pm$ '+ str(get_std(nw_accs)) + ' }'
latex_str += '& ' + str(get_avg(bc_accs)) + ' \\tiny{ $\pm$ '+ str(get_std(bc_accs)) + ' }'
latex_str += '& ' + str(get_avg(bn_accs)) + ' \\tiny{ $\pm$ '+ str(get_std(bn_accs)) + ' }'
latex_str += '& ' + str(get_avg(mz_accs)) + ' \\tiny{ $\pm$ '+ str(get_std(mz_accs)) + ' }'
latex_str += '& ' + str(get_avg(wb_accs)) + ' \\tiny{ $\pm$ '+ str(get_std(wb_accs)) + ' }'
latex_str += '& ' + str(get_avg([get_avg(nw_accs),
                                 get_avg(bc_accs),
                                 get_avg(bn_accs),
                                 get_avg(mz_accs),
                                 get_avg(wb_accs)])/100)
latex_str += '\\\\ \\hline\n'

mz_accs = []
bc_accs = []
nw_accs = []
wb_accs = []
bn_accs = []
for model in tpeonly_models:
    print(model)
    mz_accs.append(get_accuracy(model, mz_domains))
    bc_accs.append(get_accuracy(model, bc_domains))
    nw_accs.append(get_accuracy(model, nw_domains))
    wb_accs.append(get_accuracy(model, wb_domains))
    bn_accs.append(get_accuracy(model, bn_domains))
print(get_avg(mz_accs))
print(get_avg(bc_accs))
print(get_avg(nw_accs))
print(get_avg(wb_accs))
print(get_avg(bn_accs))
latex_str += '\\textbf{TE-Enc} (\\task{all}) '
latex_str += '& ' + str(get_avg(nw_accs)) + ' \\tiny{ $\pm$ '+ str(get_std(nw_accs)) + ' }'
latex_str += '& ' + str(get_avg(bc_accs)) + ' \\tiny{ $\pm$ '+ str(get_std(bc_accs)) + ' }'
latex_str += '& ' + str(get_avg(bn_accs)) + ' \\tiny{ $\pm$ '+ str(get_std(bn_accs)) + ' }'
latex_str += '& ' + str(get_avg(mz_accs)) + ' \\tiny{ $\pm$ '+ str(get_std(mz_accs)) + ' }'
latex_str += '& ' + str(get_avg(wb_accs)) + ' \\tiny{ $\pm$ '+ str(get_std(wb_accs)) + ' }'
latex_str += '& ' + str(get_avg([get_avg(nw_accs),
                                 get_avg(bc_accs),
                                 get_avg(bn_accs),
                                 get_avg(mz_accs),
                                 get_avg(wb_accs)])/100)
latex_str += '\\\\ \\hline\n'

print(latex_str)
