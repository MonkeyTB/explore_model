import tensorflow as tf
from model import NerModel
from utils import tokenize,read_vocab,format_result,read_wordvocab
import tensorflow_addons as tf_ad
from args_help import args
import json
import pandas as pd
from tqdm import tqdm
from tensorflow.python.platform import gfile
import re
from lp_pyhanlp import *

vocab2id, id2vocab = read_vocab(args.vocab_file)
tag2id, id2tag = read_vocab(args.tag_file)
embeddings_matrix, word2idx = read_wordvocab(args.fastvec_dir)
def tokenize(text, vocab2id, tag2id, word2id):
    contents = []
    labels = []
    words = []

    content = []
    label = []
    for w in text:
#             if ('\u0041' <= w <='\u005a') or ('\u0061' <= w <='\u007a'):
#                 content.append(vocab2id['<ENG>'])
        if ('0' <= w <= '9'):
            content.append(vocab2id['<NUM>'])
        else:
            content.append(vocab2id.get(w,vocab2id['<UNK>']))

    if content:
        contents.append(content)
        sententces = re.findall('[a-z0-9]+|[\u4e00-\u9fa5]+|[^a-z0-9\u4e00-\u9fa5]+',text)
        word = []
        for s in sententces:
            word.extend([j.word for j in HanLP.segment(s)])


#             word = [j.word for j in HanLP.segment(text)]
        temp = []
        for i in word:
            temp.extend([i]*len(i))
#             print(temp)
        words.append([word2id.get(i, word2id['<UNK>']) for i in temp])


    contents = tf.keras.preprocessing.sequence.pad_sequences(contents, padding='post')
    words = tf.keras.preprocessing.sequence.pad_sequences(words, padding='post')
    return contents, words
model =  tf.saved_model.load(args.savemode_dir)

content = []
degrees,educations,education_types,eq_ages,eq_workyears,gt_ages,gt_workyears,industrys,languages,lt_ages,lt_workyears,majors,school_types,skills = [],[],[],[],[],[],[],[],[],[],[],[],[],[]

online_path = r'测评/线上模型评估2.0数据返回-2021.11.1.xlsx'
df = pd.read_excel(online_path,sheet_name = '待评估数据')
for key,row in tqdm(df.iterrows()):
    content.append(row.content)
    text = row.content
    text_sequences, words_text_sequences = tokenize(text, vocab2id, tag2id, word2idx)
    logits, text_lens, log_likelihood, _ = model.call(text_sequences, words_text_sequences, text_sequences)
    paths = []
    for logit, text_len in zip(logits, text_lens):
        viterbi_path, _ = tf_ad.text.viterbi_decode(logit[:text_len], model.transition_params)
        paths.append(viterbi_path)

    entities_result = format_result(list(row.content), [id2tag[id] for id in paths[0]])
    results = eval(json.dumps(entities_result, indent=4, ensure_ascii=False))
    degree,education,education_type,eq_age,eq_workyear,gt_age,gt_workyear,industry,language,lt_age,lt_workyear,major,school_type,skill = [],[],[],[],[],[],[],[],[],[],[],[],[],[]
    for result in results:
#             print(result,type(result))
        if result['type'] == 'degree':
            if len(result['words']) > 1 and result['words'] in ['学士','硕士','博士']:
                degree.append(result['words'])
            else:
                degree.append('')
        else:
            degree.append('')
        if result['type'] == 'education':
            if len(result['words']) > 1:
                education.append(result['words'])
            else:
                education.append('')
        else:
            education.append('')
        if result['type'] == 'education_type':
            if len(result['words']) > 1 and result['words'] in ['统招','非统招','全日制','非全日制']:
                education_type.append(result['words'])
            else:
                education_type.append('')
        else:
            education_type.append('')
        if result['type'] == 'eq_age':
            if len(result['words']) > 1:
                eq_age.append(result['words'])
            else:
                eq_age.append('')
        else:
            eq_age.append('')

        if result['type'] == 'eq_workyear':
            if len(result['words']) > 1:
                eq_workyear.append(result['words'])
            else:
                eq_workyear.append('')
        else:
            eq_workyear.append('')
        if result['type'] == 'gt_age':
            if len(result['words']) > 1:
                gt_age.append(result['words'])
            else:
                gt_age.append('')
        else:
            gt_age.append('')
        if result['type'] == 'gt_workyear':
            if len(result['words']) > 1:
                gt_workyear.append(result['words'])
            else:
                gt_workyear.append('')
        else:
            gt_workyear.append('')
        if result['type'] == 'industry':
            if len(result['words']) > 1:
                industry.append(result['words'])
            else:
                industry.append('')
        else:
            industry.append('')

        if result['type'] == 'language':
            if len(result['words']) > 1:
                language.append(result['words'])
            else:
                language.append('')
        else:
            language.append('')
        if result['type'] == 'lt_age':
            if len(result['words']) > 1:
                lt_age.append(result['words'])
            else:
                lt_age.append('')
        else:
            lt_age.append('')
        if result['type'] == 'lt_workyear':
            if len(result['words']) > 1:
                lt_workyear.append(result['words'])
            else:
                lt_workyear.append('')
        else:
            lt_workyear.append('')
        if result['type'] == 'major':
            if len(result['words']) > 1:
                major.append(result['words'])
            else:
                major.append('')
        else:
            major.append('')

        if result['type'] == 'school_type':
            if len(result['words']) > 1:
                school_type.append(result['words'])
            else:
                school_type.append('')
        else:
            school_type.append('')
        if result['type'] == 'skill':
            if len(result['words']) > 1:
                skill.append(result['words'])
            else:
                skill.append('')
        else:
            skill.append('')
    degree = [i for i in set(degree) if i != '']
    degrees.append(','.join(degree))

    education = [i for i in set(education) if i != '']
    educations.append(','.join(education) )

    education_type = [i for i in set(education_type) if i != '']
    education_types.append(','.join(education_type) )

    eq_age = [i for i in set(eq_age) if i != '']
    eq_ages.append(','.join(eq_age) )

    eq_workyear = [i for i in set(eq_workyear) if i != '']
    eq_workyears.append(','.join(eq_workyear) )

    gt_age = [i for i in set(gt_age) if i != '']
    gt_ages.append(''.join(gt_age) )

    gt_workyear = [i for i in set(gt_workyear) if i != '']
    gt_workyears.append(','.join(gt_workyear) )

    industry = [i for i in set(industry) if i != '']
    industrys.append(','.join(industry) )

    language = [i for i in set(language) if i != '']
    languages.append(','.join(language) )

    lt_age = [i for i in set(lt_age) if i != '']
    lt_ages.append(','.join(lt_age) )

    lt_workyear = [i for i in set(lt_workyear) if i != '']
    lt_workyears.append(','.join(lt_workyear) )

    major = [i for i in set(major) if i != '']
    majors.append(','.join(major) )

    school_type = [i for i in set(school_type) if i != '']
    school_types.append(','.join(school_type) )

    skill = [i for i in set(skill) if i != '']
    skills.append(','.join(skill ))
c = {'content':content,'degree':degrees,'education':educations,'education_type':education_types,'eq_age':eq_ages,'eq_workyear':eq_workyears,'gt_age':gt_ages,'gt_workyear':gt_workyears,'industry':industrys,'language':languages,'lt_age':lt_ages,'lt_workyear':lt_workyears,'major':majors,'school_type':school_types,'skill':skills}
df_on = pd.DataFrame(c)
df_on.to_csv(r'测评/online.csv',encoding='utf-8')
