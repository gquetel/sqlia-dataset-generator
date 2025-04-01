import random as rd
import logging

from .load_data import CONST_INJECTION_FILES
from .load_data import parsing_injection_csv
from .load_data import parsing__benign_csv
from .load_data import set_global_var_database

from .malicious_inject import change_parameters
logger = logging.getLogger(__name__)


def select_injection(per_attack: [], nb_malicious_inject: [],  nb_queries: int):
    # selects the malicious injection, ensuring that it is all

    possible = []

    for index in range(len(per_attack)):
        if nb_malicious_inject[index] * sum(per_attack) < per_attack[index] * nb_queries:
            possible += [index]

    out = -1

    # in theory, we never get saturated
    if len(possible) != 0:
        out = possible[rd.randrange(len(possible))]
        nb_malicious_inject[out] += 1

    return out


def mixing(query: [], dirpath: str, per_attack: [], nb_queries: int, nb_malicious_query: int,
           density_malicious_query: float, file_output_safe, file_output_attack,
           per_char_obfuscation: float):
    # mix a query with possible benign and malicious injections

    # load all possible injection values for the query
    benign_inject = []
    accept_malicious_injection = []
    fix_parts = []
    malicious_inject = parsing_injection_csv()

    for index in range(len(query)):
        if index % 2:
            benign_inject += [parsing__benign_csv(dirpath, query[index])]

            if query[index][0] != '!':
                accept_malicious_injection += [int((index-1)/2)]
        else:
            fix_parts += [query[index]]

    index_nb_query = 0
    index_type_inject = (len(CONST_INJECTION_FILES))*[0]

    nb_max_benign_query_suit = 1
    for n in benign_inject:
        nb_max_benign_query_suit *= len(n)
    nb_max_benign_query = min(nb_max_benign_query_suit, nb_queries)

    if (nb_max_benign_query < nb_queries):
        print('[WARNING] The number of asked queries is greater than the number of possible different queries for :' +
              " ".join(query) + " of " + dirpath)

    if (nb_malicious_query > 0) & (len(accept_malicious_injection) == 0):
        print("[WARNING] The query : '" + " ".join(query) +
              " of " + dirpath + "' don't have inject place")
        nb_malicious_query = 0

    while index_nb_query < nb_max_benign_query + nb_malicious_query:
        query = ''
        table_select_inject = []
        malicious_query = False

        # select the different locations where malicious code will be injected
        # create all attacks first
        if (index_nb_query < nb_malicious_query):
            malicious_query = True
            table_select_inject = [
                accept_malicious_injection[rd.randrange(len(accept_malicious_injection))]]

            while (rd.random() < density_malicious_query) & (len(table_select_inject) != len(accept_malicious_injection)):
                possible = []
                for possible_inject in accept_malicious_injection:
                    if not (possible_inject in table_select_inject):
                        possible += [possible_inject]

                if len(possible) > 0:
                    table_select_inject += [
                        possible[rd.randrange(len(possible))]]

        # build the query, with the fixed parts and then the injections
        for index_injection in range(len(benign_inject)):
            query += fix_parts[index_injection]

            if index_injection in table_select_inject:
                table_select = select_injection(
                    per_attack, index_type_inject, nb_malicious_query)
                temp = rd.randrange(len(malicious_inject[table_select-1]))

                query += change_parameters(malicious_inject[table_select-1][temp], benign_inject[index_injection],
                                           per_char_obfuscation)

            else:
                temp = rd.randrange(len(benign_inject[index_injection]))
                query += benign_inject[index_injection][temp]

        query += fix_parts[len(fix_parts)-1]

        # we distribute the query in the corresponding file
        if malicious_query:
            file_output_attack.write(query)
        else:
            file_output_safe.write(query)

        index_nb_query += 1

    return (nb_max_benign_query, nb_malicious_query)


def validate_configuration(nb_generated_attack: [], per_query_type: [],  per_density_malicious_inject: float, per_char_obfuscation: float):
    """_summary_

    Args:
        nb_generated_attack (_type_): _description_
        per_query_type (_type_): _description_
        nb_generated_queries (int): _description_
        nb_malicious_query (int): _description_
        per_density_malicious_inject (float): _description_
        per_char_obfuscation (float): _description_
    """
    if sum(nb_generated_attack) > 1:
        logger.error(
            "The number of malevolence injections is greater than the number of generated queries")
        exit()

    for index in nb_generated_attack:
        if index < 0:
            logger.error(
                "A percent of type of malevolence injections is a minus number")
            exit()

    if sum(nb_generated_attack) == 0:
        logger.error(
            "All the percent of type of malevolence injections are equal to 0")
        exit()

    if len(nb_generated_attack) != len(CONST_INJECTION_FILES):
        logger.error("The number of attack types is too big")
        exit()

    if sum(per_query_type) != 1:
        logger.error("Statement reparition is not equal to 1.")
        exit()

    if per_density_malicious_inject > 1:
        logger.error("The percent of attack density is greater than one")
        exit()
    elif per_density_malicious_inject < 0:
        logger.error("The percent of attack density is lower than 0")
        exit()

    if per_char_obfuscation > 1:
        logger.error("The percent of char obfuscation is greater than one")
        exit()
    elif per_char_obfuscation < 0:
        logger.error("The percent of char obfuscation is lower than 0")
        exit()


def end_list_query(query_templates: [], dir_queries: str, list_dir_database: [],
                   dir_data: str, safe_output, attack_output, per_generated_attack: [],
                   nb_benign_query: int, nb_malicious_query: int, per_density_malicious_inject: float,
                   per_char_obfuscation: float):
    # function that mixes a list of queries one by one
    # this function will not use the appropriate percentage for the genearation
    # and will just gnerate the good number of query

    nb_still_benign_query = nb_benign_query
    nb_still_malicious_query = nb_malicious_query

    for dir_database in list_dir_database:
        set_global_var_database(dir_data+dir_database)

        for index in range(len(query_templates)):

            # load all the bases queries
            file_query = open(dir_data+dir_database +
                              dir_queries+query_templates[index], 'r')
            queries = file_query.readlines()
            file_query.close()

            for line in queries:

                # injectable parts are isolated
                separate1 = line[0:len(line)].split('}')
                separate2 = []
                for word in separate1:
                    separate2 += word.split('{')

                (nb_tmp_benign_query, nb_tmp_malicious_query) = mixing(separate2, dir_data+dir_database,
                                                                       per_generated_attack, nb_still_benign_query, nb_still_malicious_query,
                                                                       per_density_malicious_inject, safe_output, attack_output, per_char_obfuscation)

                # this permit to know how many query did we really generate
                nb_still_benign_query -= nb_tmp_benign_query
                nb_still_malicious_query -= nb_tmp_malicious_query

                # we continue until we generate all that we need
                if nb_still_benign_query == 0 & nb_still_malicious_query == 0:
                    return

    # we repeat the process if we can't generate all that we need
    end_list_query(query_templates, dir_queries, list_dir_database, dir_data, safe_output, attack_output,
                   per_generated_attack, nb_still_benign_query, nb_still_malicious_query,
                   per_density_malicious_inject, per_char_obfuscation)


def list_query(query_templates: [], per_query_type: [], dir_queries: str, list_dir_database: [],
               dir_data: str, output_file_safe: str, output_file_attack: str, per_generated_attack: [],
               nb_benign_query: int, nb_malicious_query: int, per_density_malicious_inject: float,
               per_char_obfuscation: float):
    """ Function that mixes a list of queries one by one

    Args:
        query_templates (_type_): _description_
        per_query_type (_type_): _description_
        dir_queries (str): _description_
        list_dir_database (_type_): _description_
        dir_data (str): _description_
        output_file_safe (str): _description_
        output_file_attack (str): _description_
        per_generated_attack (_type_): _description_
        nb_benign_query (int): _description_
        nb_malicious_query (int): _description_
        per_density_malicious_inject (float): _description_
        per_char_obfuscation (float): _description_
    """

    validate_configuration(per_generated_attack, per_query_type,
                           per_density_malicious_inject, per_char_obfuscation)

    # normalize the number of queries of each type to be generated according to the number of database
    nb_benign_query_per_database = float(
        nb_benign_query) / len(list_dir_database)
    nb_malicious_query_per_database = float(
        nb_malicious_query) / len(list_dir_database)

    safe_output = open(output_file_safe, 'w')
    attack_output = open(output_file_attack, 'w')

    nb_tot_benign_query = 0
    nb_tot_malicious_query = 0

    for dir_database in list_dir_database:
        set_global_var_database(dir_data+dir_database)

        for index in range(len(query_templates)):

            # load all the bases queries
            file_query = open(dir_data+dir_database +
                              dir_queries+query_templates[index], 'r')
            queries = file_query.readlines()
            file_query.close()

            # normalize the number of queries of each type to be generated according to user information
            nb_benign_query_per_file = nb_benign_query_per_database * \
                per_query_type[index] / sum(per_query_type)
            nb_malicious_query_per_file = nb_malicious_query_per_database * \
                per_query_type[index] / sum(per_query_type)

            # we normalize the number of queries to generated per queries provided
            nb_benign_queries_per_query = int(
                nb_benign_query_per_file / len(queries))
            nb_malicious_query_per_query = int(
                nb_malicious_query_per_file / len(queries))

            for line in queries:

                # injectable parts are isolated
                separate1 = line[0:len(line)].split('}')
                separate2 = []
                for word in separate1:
                    separate2 += word.split('{')

                (nb_tmp_benign_query, nb_tmp_malicious_query) = mixing(separate2, dir_data+dir_database,
                                                                       per_generated_attack, nb_benign_queries_per_query, nb_malicious_query_per_query,
                                                                       per_density_malicious_inject, safe_output, attack_output, per_char_obfuscation)

                # this permit to know how many query did we really generate
                nb_tot_benign_query += nb_tmp_benign_query
                nb_tot_malicious_query += nb_tmp_malicious_query

    # We can't always generate all the queries ask,
    # because of that the percent cannot divide properly the nulber of queries ask
    # because of the data that cannot generate a suffisant number of unique queries

    nb_still_benign_query = nb_benign_query - nb_tot_benign_query
    if nb_still_benign_query > 0:
        print("[WARNING] " + str(nb_still_benign_query) +
              " benign queries can not have been generated proprely")

    nb_still_malicious_query = nb_malicious_query - nb_tot_malicious_query
    if nb_still_malicious_query > 0:
        print("[WARNING] " + str(nb_still_malicious_query) +
              " malicious queries can not have been generated proprely")

    # use to generate the last queries that we couldn't generate properly
    end_list_query(query_templates, dir_queries, list_dir_database, dir_data, safe_output, attack_output,
                   per_generated_attack, nb_still_benign_query, nb_still_malicious_query,
                   per_density_malicious_inject, per_char_obfuscation)

    safe_output.close()
    attack_output.close()
