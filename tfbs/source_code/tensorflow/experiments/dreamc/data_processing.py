import time
import numpy as np
import os
import sys

dreamc_nucleotide_dict = {'N':0, 'A':1, 'T':2, 'G':3, 'C':4}

def create_aa_dict():
    print 'started create_aa_dict method'
    aa_dict = {}
    start_time = time.clock()
    aa_seq_2045 = 'MAGRSHPGPLRPLLPLLVVAACVLPGAGGTCPERALERREEEANVVLTGTVEEILNVDPVQHTYSCKVRVWRYLKGKDLVARESLLDGGNKVVISGFGDPLICDNQVSTGDTRIFFVNPAPPYLWPAHKNELMLNSSLMRITLRNLEEVEFCVEDKPGTHFTPVPPTPPDACRGMLCGFGAVCEPNAEGPGRASCVCKKSPCPSVVAPVCGSDASTYSNECELQRAQCSQQRRIRLLSRGPCGSRDPCSNVTCSFGSTCARSADGLTASCLCPATCRGAPEGTVCGSDGADYPGECQLLRRACARQENVFKKFDGPCDPCQGALPDPSRSCRVNPRTRRPEMLLRPESCPARQAPVCGDDGVTYENDCVMGRSGAARGLLLQKVRSGQCQGRDQCPEPCRFNAVCLSRRGRPRCSCDRVTCDGAYRPVCAQDGRTYDSDCWRQQAECRQQRAIPSKHQGPCDQAPSPCLGVQCAFGATCAVKNGQAACECLQACSSLYDPVCGSDGVTYGSACELEATACTLGREIQVARKGPCDRCGQCRFGALCEAETGRCVCPSECVALAQPVCGSDGHTYPSECMLHVHACTHQISLHVASAGPCETCGDAVCAFGAVCSAGQCVCPRCEHPPPGPVCGSDGVTYGSACELREAACLQQTQIEEARAGPCEQAECGSGGSGSGEDGDCEQELCRQRGGIWDEDSEDGPCVCDFSCQSVPGSPVCGSDGVTYSTECELKKARCESQRGLYVAAQGACRGPTFAPLPPVAPLHCAQTPYGCCQDNITAARGVGLAGCPSACQCNPHGSYGGTCDPATGQCSCRPGVGGLRCDRCEPGFWNFRGIVTDGRSGCTPCSCDPQGAVRDDCEQMTGLCSCKPGVAGPKCGQCPDGRALGPAGCEADASAPATCAEMRCEFGARCVEESGSAHCVCPMLTCPEANATKVCGSDGVTYGNECQLKTIACRQGLQISIQSLGPCQEAVAPSTHPTSASVTVTTPGLLLSQALPAPPGALPLAPSSTAHSQTTPPPSSRPRTTASVPRTTVWPVLTVPPTAPSPAPSLVASAFGESGSTDGSSDEELSGDQEASGGGSGGLEPLEGSSVATPGPPVERASCYNSALGCCSDGKTPSLDAEGSNCPATKVFQGVLELEGVEGQELFYTPEMADPKSELFGETARSIESTLDDLFRNSDVKKDFRSVRLRDLGPGKSVRAIVDVHFDPTTAFRAPDVARALLRQIQVSRRRSLGVRRPLQEHVRFMDFDWFPAFITGATSGAIAAGATARATTASRLPSSAVTPRAPHPSHTSQPVAKTTAAPTTRRPPTTAPSRVPGRRPPAPQQPPKPCDSQPCFHGGTCQDWALGGGFTCSCPAGRGGAVCEKVLGAPVPAFEGRSFLAFPTLRAYHTLRLALEFRALEPQGLLLYNGNARGKDFLALALLDGRVQLRFDTGSGPAVLTSAVPVEPGQWHRLELSRHWRRGTLSVDGETPVLGESPSGTDGLNLDTDLFVGGVPEDQAAVALERTFVGAGLRGCIRLLDVNNQRLELGIGPGAATRGSGVGECGDHPCLPNPCHGGAPCQNLEAGRFHCQCPPGRVGPTCADEKSPCQPNPCHGAAPCRVLPEGGAQCECPLGREGTFCQTASGQDGSGPFLADFNGFSHLELRGLHTFARDLGEKMALEVVFLARGPSGLLLYNGQKTDGKGDFVSLALRDRRLEFRYDLGKGAAVIRSREPVTLGAWTRVSLERNGRKGALRVGDGPRVLGESPVPHTVLNLKEPLYVGGAPDFSKLARAAAVSSGFDGAIQLVSLGGRQLLTPEHVLRQVDVTSFAGHPCTRASGHPCLNGASCVPREAAYVCLCPGGFSGPHCEKGLVEKSAGDVDTLAFDGRTFVEYLNAVTESEKALQSNHFELSLRTEATQGLVLWSGKATERADYVALAIVDGHLQLSYNLGSQPVVLRSTVPVNTNRWLRVVAHREQREGSLQVGNEAPVTGSSPLGATQLDTDGALWLGGLPELPVGPALPKAYGTGFVGCLRDVVVGRHPLHLLEDAVTKPELRPCPTP'
    aa_unique_2045 = list(set(aa_seq_2045.upper()))
    aa_dict['X'] = 0
    for A in aa_unique_2045:
        aa_dict[A] = aa_unique_2045.index(A)+1
    print 'Number of unique AA in dreamc_aa_dict are %d' %len(aa_unique_2045)
    print 'completed create_aa_dict method, Time Taken %s' % str(time.clock() - start_time)
    return aa_dict

def create_annotation_dict(data_dir, annot_filename):
    print 'started create_annotation_dict method'
    start_time = time.clock()
    data = {}
    with open(data_dir+'/'+annot_filename) as f:
        for line in [l.rstrip('\n') for l in f.readlines()]:
            (key, start, end) = line.split('\t')
            if data.has_key(key):
                data[key].append((start, end))
            else:
                data[key] = []
                data[key].append((start, end))
    print 'completed create_annotation_dict method, Time Taken %s' % str(time.clock() - start_time)
    return data

def create_dnase_data(data_dir, dnase_filenames, annot_filename):
    print 'started update_dnase_data method'
    start_time = time.clock()
    dnase_path = data_dir +'/dnase/'+dnase_filenames
    annot_dict = create_annotation_dict(data_dir, annot_filename)

    with open(dnase_path) as f:
        bio_f = f.readline().split('\n')[0]
        while bio_f != '':
            bio_s = bio_f.split('.')[1]
            print 'processing filename %s' % bio_f
            start_time_f = time.clock()
            prev_chr = ''
            list = []
            chr_file_name = ''
            with open(data_dir+'/dnase/'+bio_f) as bf:
                data = bf.readline().split('\n')[0]
                while data!='':
                    data = data.split('\t')
                    chr = data[0]
                    if chr in annot_dict.keys():
                        if chr_file_name == '':
                            chr_file_name = data_dir + '/processed_data/dnase_npy/' + bio_s + '.' + chr + '.npy'
                        if list == []:
                            list = [0]*int(annot_dict[chr][-1][1])

                        if not os.path.isfile(chr_file_name):
                            if prev_chr != '' and prev_chr != chr:
                                np.save(chr_file_name,np.array(list))
                                list = [0] * int(annot_dict[chr][-1][1])
                                chr_file_name = data_dir + '/processed_data/dnase_npy/' + bio_s + '.' + chr + '.npy'

                            for i in range(int(data[1]),int(data[2])):
                                if i < len(list):
                                    list[i] = float(data[3])
                        else:
                            if prev_chr != '' and prev_chr != chr:
                                list = [0] * int(annot_dict[chr][-1][1])
                                chr_file_name = data_dir + '/processed_data/dnase_npy/' + bio_s + '.' + chr + '.npy'
                                for i in range(int(data[1]), int(data[2])):
                                    if i < len(list):
                                        list[i] = float(data[3])
                        prev_chr = chr
                    else:
                        if chr_file_name != '':
                            np.save(chr_file_name, np.array(list))
                        prev_chr = ''
                        chr_file_name = ''
                        list = []
                    data = bf.readline().split('\n')[0]
                if not os.path.isfile(chr_file_name):
                    np.save(chr_file_name, np.array(list))
            print 'completed processing file... , Time Taken %s' % str(time.clock() - start_time_f)
            bio_f = f.readline().split('\n')[0]
    print 'completed update_dnase_data method, Time Taken %s' %str(time.clock()-start_time)

def create_genome_data(data_dir, genome_filename, annot_filename):
    print 'started create_genome_data method'
    start_time = time.clock()
    annot_dict = create_annotation_dict(data_dir, annot_filename)

    chrs = annot_dict.keys()
    with open(data_dir+'/genome/'+genome_filename) as f:
        line = f.readline().split('\n')[0]
        index = 0
        list = []
        chr_file_name = ''
        while line != '':
            if line[0] == '>':
                chr = line[1:]
                if chr_file_name != '':
                    np.save(chr_file_name,np.array(list,dtype ='uint8'))
                if chr in chrs:
                    chr_file_name = data_dir + '/processed_data/genome_npy/hg19.' + chr + '.npy'
                    list = [0]*int(annot_dict[chr][-1][1])
                    index = 0
                else:
                    chr_file_name = ''
            elif chr_file_name != '':
                if index < len(list):
                    for N in line:
                        int_nucleotide = dreamc_nucleotide_dict[N.upper()]
                        list[index] = int_nucleotide
                        index = index + 1
            line = f.readline().split('\n')[0]
        if chr_file_name != '':
            np.save(chr_file_name, np.array(list, dtype='uint8'))
    print 'completed create_genome_data method, Time Taken %s' % str(time.clock() - start_time)

def create_tf_aa_seq(data_dir, aa_filename, tf_array, dreamc_aa_dict):
    print 'started update_tf_aa_seq method'
    start_time = time.clock()

    tf_n = ''
    aa_list = []
    aa_dict = {}

    with open(data_dir+'/'+aa_filename) as f:
        line = f.readline().split('\n')[0]
        while line != '':
            if line[0] == '>':
                if not aa_list == []:
                    if aa_dict.has_key(tf_n):
                        if len(aa_list) > len(aa_dict[tf_n]):
                            aa_dict[tf_n] = aa_list
                    else:
                        aa_dict[tf_n] = aa_list
                aa_list = []
                tf_n = line.split('|')[6]
            elif tf_n != '' and tf_n not in tf_array:
                pass
            elif tf_n != '':
                for A in line:
                    aa_list.append(dreamc_aa_dict[A.upper()])
            line = f.readline().split('\n')[0]
    print 'completed update_tf_aa_seq method, Time Taken %s' % str(time.clock() - start_time)
    return aa_dict

def process_labels_file(data_dir,labels_filenames):
    labels_path = data_dir + '/labels/' + labels_filenames
    with open(labels_path) as f:
        tf_fn = f.readline().split('\n')[0]
        while tf_fn != '':
            tf_f = open(data_dir + '/labels/' + tf_fn)
            tf_n = tf_fn.split('.')[0]
            header = tf_f.readline().split('\n')[0]
            bio_samples = header.split('\t')[3:]
            fn_handle_dict = {}
            handle = open(data_dir + '/processed_data/U/'+tf_n+'.'+bio_samples[0]+'.U.tsv','a')
            fn_handle_dict[tf_n+'.'+bio_samples[0]+'.U.tsv'] = handle
            data = tf_f.readline().split('\n')[0]
            while data != '':
                data_split = data.split('\t')
                labels = data_split[3:]
                index = 0
                for label in labels:
                    if label == 'U':
                        key = tf_n+'.'+bio_samples[index]+'.U.tsv'
                        if key in fn_handle_dict:
                            handle = fn_handle_dict[key]
                        else:
                            handle = open(data_dir + '/processed_data/U/'+ key,'a')
                            fn_handle_dict[key] = handle

                    elif label == 'A':
                        key = tf_n+'.'+bio_samples[index]+'.A.tsv'
                        if key in fn_handle_dict:
                            handle = fn_handle_dict[key]
                        else:
                            handle = open(data_dir + '/processed_data/A/' + key, 'a')
                            fn_handle_dict[key] = handle

                    elif label == 'B':
                        key = tf_n+'.'+bio_samples[index]+'.B.tsv'
                        if key in fn_handle_dict:
                            handle = fn_handle_dict[key]
                        else:
                            handle = open(data_dir + '/processed_data/B/' + key, 'a')
                            fn_handle_dict[key] = handle

                    line = '\t'.join(data_split[0:3])
                    line = line+'\t'+label+'\n'
                    handle.write(line)
                    index = index + 1
                data = tf_f.readline().split('\n')[0]

            tf_f.close()
            tf_fn = f.readline().split('\n')[0]

def process_sample_data():
    data_dir = '/home/rohitjain/Projects/TFBS/repo/level_0-pipeline/dataset/sample_dataset'
    annot_filename = 'annot_file'
    labels_filenames = 'labels_filenames'
    dnase_filenames = 'dnase_filenames'
    genome_filename = 'genomehg19'
    aa_filename = 'aa_seq.fa'
    tfs = ['ATF3', 'ATF7']

    aa_dict = create_aa_dict()
    create_dnase_data(data_dir, dnase_filenames, annot_filename)
    create_genome_data(data_dir, genome_filename, annot_filename)
    tf_aa_dict = create_tf_aa_seq(data_dir, aa_filename, tfs, aa_dict)
    process_labels_file(data_dir, labels_filenames)

def sample_data_dict():

    data_dir = '/home/rohitjain/Projects/TFBS/repo/level_0-pipeline/dataset/sample_dataset'
    aa_dict = create_aa_dict()
    aa_filename = 'aa_seq.fa'
    tfs = ['ATF3', 'ATF7']

    sample_data_dict = {
        'data_dir' : data_dir,
        'max_queue_size_per_class' : 1000,
        'max_enqueue_buffer_per_class': 100,
        'save_after_steps': 100,
        'summary_after_steps':10,
        'n_classes' : 3,
        'class_type_index_dict' : {'U': 0, 'A': 1, 'B': 2},
        'nucleotide_dict' : {'N':0, 'A':1, 'T':2, 'G':3, 'C':4},
        'flanking_region': 400,
        'bio_tf_train': {'GM12878': ['ATF3', 'ATF7'], 'A549': ['ATF3', 'ATF7']},
        'bio_tf_test': {'H1-hESC': ['ATF3', 'ATF7']},
        'bio_samples': ['A549', 'GM12878', 'H1-hESC'],
        'tfs': ['ATF3', 'ATF7'],
        'tf_aa_dict': create_tf_aa_seq(data_dir, aa_filename, tfs, aa_dict),
        'ground_truth':{'A': [data_dir+'/processed_data/A/ATF3.GM12878.A.tsv',
                              data_dir + '/processed_data/A/ATF3.A549.A.tsv',
                              data_dir + '/processed_data/A/ATF7.GM12878.A.tsv',
                              data_dir + '/processed_data/A/ATF7.A549.A.tsv'],
                        'B': [data_dir+'/processed_data/B/ATF3.GM12878.B.tsv',
                              data_dir + '/processed_data/B/ATF3.A549.B.tsv',
                              data_dir + '/processed_data/B/ATF7.GM12878.B.tsv',
                              data_dir + '/processed_data/B/ATF7.A549.B.tsv'],
                        'U': [data_dir+'/processed_data/U/ATF3.GM12878.U.tsv',
                              data_dir + '/processed_data/U/ATF3.A549.U.tsv',
                              data_dir + '/processed_data/U/ATF7.GM12878.U.tsv',
                              data_dir + '/processed_data/U/ATF7.A549.U.tsv']
                        },
        'test_data': {'A': [data_dir+'/processed_data/A/ATF3.H1-hESC.A.tsv',
                            data_dir + '/processed_data/A/ATF7.H1-hESC.A.tsv'],
                      'B': [data_dir+'/processed_data/B/ATF3.H1-hESC.B.tsv',
                            data_dir + '/processed_data/B/ATF7.H1-hESC.B.tsv'],
                      'U': [data_dir+'/processed_data/U/ATF3.H1-hESC.U.tsv',
                            data_dir + '/processed_data/U/ATF7.H1-hESC.U.tsv']}

    }

    return sample_data_dict

def test():
    data_dir = '/home/rohitjain/Projects/TFBS/repo/level_0-pipeline/dataset/sample_dataset'
    print np.load(data_dir + '/processed_data/dnase_npy/GM12878.chr10.npy').item(400)
    print np.load(data_dir + '/processed_data/dnase_npy/GM12878.chr10.npy').item(1600)
    print np.load(data_dir + '/processed_data/dnase_npy/GM12878.chrX.npy').item(1600)
    print np.load(data_dir + '/processed_data/dnase_npy/H1-hESC.chr11.npy').item(900)
    print np.load(data_dir + '/processed_data/dnase_npy/H1-hESC.chr11.npy').item(1100)
    print np.load(data_dir + '/processed_data/dnase_npy/H1-hESC.chrX.npy').item(1799)
    print np.load(data_dir + '/processed_data/dnase_npy/H1-hESC.chrX.npy').item(1800)
    print np.load(data_dir + '/processed_data/genome_npy/hg19.chrX.npy').item(1999)
    print np.load(data_dir + '/processed_data/genome_npy/hg19.chrX.npy').item(1998)
    print np.load(data_dir + '/processed_data/genome_npy/hg19.chrX.npy').item(1)
    print np.load(data_dir + '/processed_data/genome_npy/hg19.chrX.npy').item(2)

if __name__ == "__main__":
    if len(sys.argv) == 1:
        #test()
        process_sample_data()
    else:
        print 'Error! Many arguements!'