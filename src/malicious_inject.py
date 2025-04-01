import string
import random as rd

from .load_data import CONST_BOUNDARIE
from .load_data import CONST_LINES_COMMENT
from .load_data import CONST_INFERENCE_SYMBOLE
from .load_data import CONST_QUERY
from .load_data import get_GLOBAL_DATABASE_USER_NAME
from .load_data import get_GLOBAL_DATABASE_USER_NAME
from .load_data import get_GLOBAL_DATABASE_TABLE_NAME


def obfuscation (injection : str, per_char_obfuscation : float) :
    out = ""

    for char in injection :
        if rd.random() > per_char_obfuscation :
            out += char
        else :
            out += '%'+str(char.encode().hex())

    return out

def value (key : str, dic : {}, per_char_obfuscation : float) :
    # manages the dictionary of generated values 
    # be injected into malicious injections

    if not(key in dic.keys()) :
        if key[0:7] == 'RANDNUM' :
            dic[key] = str(rd.randrange(10000))

        elif key[0:9] == 'SLEEPTIME' :
            dic[key] = str(rd.randrange(10000))

        elif key[0:7] == 'RANDSTR' :
            dic[key] = ''.join(rd.choice(string.ascii_uppercase + string.digits) for _ in range(rd.randrange(20)+1))

        elif key=='GENERIC_SQL_COMMENT' :
            out = CONST_LINES_COMMENT[rd.randrange(len(CONST_LINES_COMMENT))]
            return out[0:len(out)-1]

        # delimiters are just used to easily analyze the error return
        elif key=='DELIMITER_START' :
            dic['DELIMITER_START'] =  ''.join(rd.choice(string.ascii_uppercase + string.digits) for _ in range(rd.randrange(6)+1))

        elif key=='DELIMITER_STOP' :
            dic['DELIMITER_STOP'] =  ''.join(rd.choice(string.ascii_uppercase + string.digits) for _ in range(rd.randrange(6)+1))

        elif (key[0:9] == 'ORIGVALUE') | (key[0:8] == 'ORIGINAL'):
            random_int = str(rd.randrange(10000))

            while (random_int in dic.values()):
                random_int = str(rd.randrange(10000))
            
            dic[key] = str(random_int)
            
        elif key == 'INFERENCE':
            dic[key] = str(rd.randrange(10000))
            dic[key]+= CONST_INFERENCE_SYMBOLE[rd.randrange(len(CONST_INFERENCE_SYMBOLE))]
            dic[key] = str(rd.randrange(10000))

        elif key =='QUERY' :
            dic[key] =change_parameters( CONST_QUERY[rd.randrange(len(CONST_QUERY))],[], per_char_obfuscation)

        elif key[0:6] =='COLUMN':
            num_start = rd.randrange(48)+1
            num_end = rd.randrange(50-num_start)+num_start+1

            if (key[7:9]!='10'):
                num_start =0
                num_end = 10
            
            elif (key[7:9]!='20'):
                num_start =10
                num_end = 20

            elif (key[7:9]!='30'):
                num_start =20
                num_end = 30

            elif (key[7:9]!='40'):
                num_start =30
                num_end = 40

            else :
                print("[ERROR] Attack file the ["+ key + "] value is not known")
                exit()

            dic[key]=''
            for index in range(num_end) : 

                if index < num_start:
                    dic[key]+='NULL'
                elif key[10:13]=='CHA':
                    dic[key]+="'"+rd.choice(string.ascii_uppercase + string.digits)+"'"
                elif key [10:13]=='NUL':
                    dic[key]+='NULL'
                elif key [10:13]=='NUM':
                    dic[key]+= str(rd.randrange(9))
                else :
                    print("[ERROR] Attack file the ["+ key + "] value is not known")
                    exit()
                
                if (index!=num_end-1):
                    dic[key]+=', '
        
        elif key =='!table_name' :
            return get_GLOBAL_DATABASE_TABLE_NAME()[rd.randrange(len(get_GLOBAL_DATABASE_TABLE_NAME()))]
        
        elif key =='!user_name' :
            return get_GLOBAL_DATABASE_USER_NAME()[rd.randrange(len(get_GLOBAL_DATABASE_USER_NAME()))]
        
        elif key =='!column_name' :
            return get_GLOBAL_DATABASE_USER_NAME()[rd.randrange(len(get_GLOBAL_DATABASE_USER_NAME()))]
        
        else :
            print("[ERROR] Attack file the ["+ key + "] value is not known")
            exit()
    
    return obfuscation(dic[key], per_char_obfuscation)

def select_boundarie (where : str, possible_inject : [], per_char_obfuscation : float) :
    # is used to select a prefix and suffix in random ways

    num = where[5]
    possible_choice = []

    for subtab in CONST_BOUNDARIE :
        if num in subtab [0] : 
            possible_choice+=[subtab]

    choice = list(possible_choice[rd.randrange(len(possible_choice))])
    inject =''

    # a necessary answer is added, either valid or inaccurate
    if num =='1' :
        inject = possible_inject [rd.randrange(len(possible_inject))]
    elif num == '2' :
        inject = str (- rd.randrange(1000))
    
    choice[1] = inject + change_parameters(choice[1],[], per_char_obfuscation)
    choice[2] = change_parameters(choice[2][:len(choice[2])-1],[], per_char_obfuscation) # the -1 is used to erase the \n

    return choice

def change_parameters (inject :str, possible : [], per_char_obfuscation : float):
    # change the injectable parts of malicious injections

    # isolate the injectable parts
    separate1 = inject.split('}')
    separate2 = []
    for word in separate1:
        separate2 += word.split('{')

    query_traduct =separate2[0]
    query_non_traduct = separate2[:len(separate2)-1]
    boundarie=[-1,'','']
    dic = {}

    if len(separate2) > 1 :

        # if there's an inject of 'where' it will always be in the second place
        # this determines the prefic and suffix of the injection 
        if (separate2[1][:5]=='WHERE') & (possible!=[]):
            boundarie = select_boundarie (separate2[1], possible, per_char_obfuscation)
            query_non_traduct = separate2 [2:len(separate2)-1]
            
        query_traduct+= boundarie[1]            

        for index in range(len(query_non_traduct)) :
            if index%2 :
                query_traduct+=value(query_non_traduct[index], dic, per_char_obfuscation)
            else :
                query_traduct+=query_non_traduct[index]
        

        query_traduct += boundarie[2] 
    return query_traduct