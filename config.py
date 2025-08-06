GRAPH_TEMPLATE={
    'desc':{
        'slots':['Disease'],
        'question':'什么叫%Disease%?/%Disease%是一种什么病?',
        'cypher':"MATCH(n:Disease) WHERE n.name='%Disease%' RETURN n.desc AS RES",
        'answer':'[%Disease%]的定义:%RES%',
    },
    'cause':{
        'slots':['Disease'],
        'question':'%Disease%一般是由什么引起的?/什么会导致%Disease%?',
        'cypher':"MATCH(n:Disease) WHERE n.name='%Disease%' RETURN n.cause AS RES",
        'answer':'[%Disease%]的病因:%RES%'
    },
    'Disease_Symptom':{
        'slots':['Disease'],
        'question':'%Disease%会有哪些症状?/%Disease%有哪些临床表现?',
        'cypher':"MATCH(n:Disease)-[:DISEASE_SYMPTOM]->(m)WHERE n.name='%Disease%'RETURN SUBSTRING(REDUCE(s='',x IN COLLECT(m.name)ls +'+ x), 1) AS RES",
        'answer':'[%Disease%]的症状:%RES%'
    },
    'Symptom':{
        'slots':['Symptom'],
        'question':'%Symptom%可能是得了什么病?',
        'cypher':"MATCH(n)-[:DISEASE_SYMPTOM]->(m:Symptom) WHERE m.name='%Symptom%'RETURN SUBSTRING(REDUCE(s='',x IN COLLECT(n.name)s+'+x),1)AS RES",
        'answer':'可能出现[%Symptom%]症状的疾病:%RES%'
    },
    'cure_way':{
        'slots':['Disease'],
        'question':'%Disease%吃什么药好得快?/%Disease%怎么治?',
        'cypher':'''
            MATCH (n:Disease)-[:DISEASE_CUREWAY]->(m1),
            (n:Disease)-[:DISEASE_DRUG]->(m2),
            (n:Disease)-[:DISEASE_DO_EAT]->(m3) 
            WHERE n.name ='%Disease%'
            WITH COLLECT(DISTINCT m1.name) AS m1Names,
                COLLECT(DISTINCT m2.name) AS m2Names,
                COLLECT(DISTINCT m3.name) AS m3Names 
            RETURN SUBSTRING(REDUCE(s ='',x IN m1Names |s+'、'+ x),1) AS RES1, 
                SUBSTRING(REDUCE(s='', x IN m2Names |s+'、'+x),1) AS RES2,
                SUBSTRING(REDUCE(s='', x IN m3Names |s+'、'+x),1) AS RES3
            ''',
        'answer':'[%Disease%]的治疗方法:%RES1%。\n可用药物:%RES2%。\n推荐食物:%RES3%'
    },
    'cure_department':{
        'slots':['Disease'],
        'question':'得了%Disease%去医院挂什么科室的号?',
        'cypher':"MATCH(n:Disease)-[:DISEASE_DEPARTMENT]->(m)WHERE n.name='%Disease%'RETURN SUBSTRING(REDUCE(s='',x IN COLLECT(m.name)|s+'、'+x),1)AS RES",
        'answer':'[%Disease%]的就诊科室:%RES%'
    },
    'prevent':{
        'slots':['Disease'],
        'question':'%Disease%要怎么预防?',
        'cypher':"MATCH(n:Disease)WHERE n.name='%Disease%' RETURN n.prevent AS RES",
        'answer':'[%Disease%]的预防方法:%RES%'
    },
    'not_eat':{
        'slots':['Disease'],
        'question':'%Disease%换着有什么禁忌?/%Disease%不能吃什么?',
        'cypher':"MATCH(n:Disease)-[:DISEASE_NOT_EAT]->(m)WHERE n.name='%Disease%'RETURN SUBSTRING(REDUCE(s='',x IN COLLECT(m.name), s+'、'+ x),1)AS RES",
        'answer':'[%Disease%]的患者不能吃的食物:%RES%'
    },
    'check':{
        'slots':['Disease'],
        'question':'%Disease%要做哪些检查?',
        'cypher':"MATCH(n:Disease)-[:DISEASE_CHECK]->(m)WHERE n.name='%Disease%'RETURN SUBSTRING(REDUCE(s='',x IN COLLECT(m.name)|s+'、'+x),1) AS RES",
        'answer':'[%Disease%]的检查项目:%RES%'
    },
    'cured_prob':{
        'slots':['Disease'],
        'question':'%Disease%能治好吗?/%Disease%治好的几率有多大?',
        'cypher':"MATCH(n:Disease) WHERE n.name='%Disease%' RETURN n.cured_prob AS RES",
        'answer':'[%Disease%]的治愈率:%RES%'
    },
    # 'acompany':{
    #     'slots':['Disease'],
    #     'question':'%Disease%的并发症有哪些?',
    #     'cypher':"MATCH(n:Disease)-[:DISEASE_ACOMPANY]->(m)WHERE n.name='%Disease%'RETURN SUBSTRING(REDUCE(s='',x IN COLLECT(m.name)|s+'、'+ x),1)AS RES",
    #     'answer':'[%Disease%]的并发症:%RES%'
    # },
    'indications':{
        'slots': ['Drug'],
        'question':'%Drug%能治那些病?',
        'cypher':"MATCH(n:Disease)-[:DISEASE_DRUG]->(m:Drug)WHERE m.name='%Drug%'RETURN SUBSTRING(REDUCE(s='',x IN COLLECT(n.name)|s+'、' +x),1)AS RES",
        'answer':'[%Drug%]能治疗的疾病有:%RES%'
    }
}
