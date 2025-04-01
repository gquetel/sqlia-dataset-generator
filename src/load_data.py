# list des fichiers injectables 
CONST_INJECTION_DIR = './attacker/'
CONST_INJECTION_FILES = ['boolean','error','inline','stacked','time','union']

CONST_COMMENT_FILE = './attacker/typic_comment'
CONST_query_FILE = './attacker/query'
CONST_BOUNDARIE_FILE = './attacker/boundarie'

CONST_COLUMN_NAME_FILE = 'column_name'
CONST_USER_NAME_FILE = 'database_user_name'
CONST_TABLE_NAME_FILE = 'table_name'

# information that the malicious injection need to mute the attack
file = open(CONST_COMMENT_FILE,'r')
CONST_LINES_COMMENT = file.readlines()
file.close()

file = open(CONST_query_FILE, 'r')
tab = []
for line in file.readlines() :
    tab += [line[:len(line)-1]] #  the -1 is used to erase the \n
    
CONST_QUERY= tab
file.close()

file = open(CONST_BOUNDARIE_FILE, 'r')
tab = []
for line in file.readlines() :
    tab += [line.split(';')]
CONST_BOUNDARIE = tab 
file.close()


# information that the malicious injection need to attack a specific database
GLOBAL_DATABASE_COLUMN_NAME = []
GLOBAL_DATABASE_USER_NAME   = []
GLOBAL_DATABASE_TABLE_NAME  = []

# list of mysql inference symbole
CONST_INFERENCE_SYMBOLE = ['=','!=','>','>=','<','<=']

def set_global_var_database (data_dir : str) :
    #set the global list of specific database characteristic 

    global GLOBAL_DATABASE_COLUMN_NAME
    global GLOBAL_DATABASE_USER_NAME
    global GLOBAL_DATABASE_TABLE_NAME

    file = open(data_dir + CONST_COLUMN_NAME_FILE, 'r')
    GLOBAL_DATABASE_COLUMN_NAME = []
    for line in file :
        GLOBAL_DATABASE_COLUMN_NAME += [line[0:len(line)-1]] #  the -1 is used to erase the \n
    file.close()

    file = open(data_dir + CONST_USER_NAME_FILE, 'r')
    GLOBAL_DATABASE_USER_NAME   = []
    for line in file :
        GLOBAL_DATABASE_USER_NAME += [line[0:len(line)-1]] #  the -1 is used to erase the \n
    file.close()

    file = open(data_dir + CONST_TABLE_NAME_FILE, 'r')
    GLOBAL_DATABASE_TABLE_NAME  = []
    for line in file :
        GLOBAL_DATABASE_TABLE_NAME += [line[0:len(line)-1]] #  the -1 is used to erase the \n
    file.close()

def get_GLOBAL_DATABASE_TABLE_NAME () :
    return GLOBAL_DATABASE_TABLE_NAME

def get_GLOBAL_DATABASE_COLUMN_NAME () :
    return GLOBAL_DATABASE_COLUMN_NAME

def get_GLOBAL_DATABASE_USER_NAME () :
    return GLOBAL_DATABASE_USER_NAME

def parsing_injection_csv():
    # loads all malicious file values into an array of arrays

    output =[]

    for filename in CONST_INJECTION_FILES : 
        file = open(CONST_INJECTION_DIR+filename,'r')
        output.append( file.readlines())
        file.close()

    for index1 in range(len(output)):
        for index2 in range(len(output[index1])):
            output[index1][index2]= output[index1][index2][:len(output[index1][index2])-1] # the -1 is used to erase the \n

    return output
    


def parsing__benign_csv (dirpath :str, filename :str):
    # loads all values from a file into a table

    # we don't reload data that we've already had to load to make injections
    if (filename=='column_name'):
        return GLOBAL_DATABASE_COLUMN_NAME
    if (filename=='table_name'):
        return GLOBAL_DATABASE_TABLE_NAME
    if (filename=='user_name'):
        return GLOBAL_DATABASE_USER_NAME

    output = []
    file = dirpath+filename

    if filename[0]=='!':
        file =  dirpath + filename[1:]

    file = open(file,'r')
    for line in file :
        output += [line[0:len(line)-1]] #  the -1 is used to erase the \n
    file.close()

    nb_exec = len(output)
    if nb_exec == 0 :
        print("[ERROR] No value in the dir :" + dirpath+filename)
        exit()

    return output