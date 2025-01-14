import sqlite3
import time
import queue
import threading


class DatabaseSingleton:
    def __new__(cls, *args, **kwargs):
        if not hasattr(cls, '_singleton'):
            # first call
            cls._singleton = super().__new__(cls)
            # already exist        
        return cls._singleton


    def __init__(self, db_name="general_database") -> None:
        if not hasattr(self, '_initialized'):
            self._initialized = True
            self.db_name = db_name
            self.max_try = 5  # max number of try to write in the db for one process
            self.column = 7  # number of columns in my table
            self.connection = sqlite3.connect(db_name)  # it creates the db if it does not exist
            self.cursor = self.connection.cursor()  # it creates the cursor to execute the queries
            self.create_table()  # it creates the table if it does not exist
            self.write_queue = queue.Queue()  # it creates the queue to write in the db
            self.start_writer()  # it starts the writer process
            # self.close_all() # it closes the connection and the cursor, don't know if it is necessary

    def create_table(self):
        self.cursor.execute(f"CREATE TABLE IF NOT EXISTS {self.db_name}(gap, location, people_in, people_out, unique_people, date, hour)")

    def close_all(self):
        self.cursor.close()
        self.connection.close()

    # WRITER FUNCTIONS

    def start_writer(self):
        def writer():
            i = -1
            connection = sqlite3.connect(self.db_name)
            cursor = connection.cursor()
            while True:
                data = self.write_queue.get()  # Wait for a write request
                # with open("writer_log.txt", "a") as f:
                #     f.write(f"writing {data} in the database {self.db_name}\n")
                if data is None:  # None is our signal to stop the writer
                    print("while loop stopped")
                    break # EXIT FROM WHILE LOOP
                # write one data in the table
                for i in range(self.max_try):
                    try:
                        cursor.execute(f"INSERT INTO {self.db_name} VALUES(?, ?, ?, ?, ?, ?, ?)", data)
                        connection.commit()
                        break

                    except Exception as e:
                        print(f"An error occurred: {e}, retrying...\n")
                        time.sleep(0.1)
                        continue
                if i == self.max_try - 1:
                    with open("log_error.txt", "a") as f:
                        f.write(f"Error in writing {data} in the database {self.db_name}")
                    # I don't stop the writer process, because it is not a critical error and the other data can be still written

            self.write_queue.task_done()  # Signal that the write request is done
            connection.close()  # Close the connection
            cursor.close()  # Close the cursor

        writer_thread = threading.Thread(group=None, target=writer, name="writer_process", daemon=True)
        writer_thread.start()  # Start the writer thread, so it starts the writer process in background

    def write_to_database(self, data, choise: bool):
        if choise:
            self.write_multiple(data)
        else:
            self.write_one(data)

    def write_one(self, data: list):
        data = self.standardize_data(data)
        # for i in range(len(data)):
        #     if isinstance(data[i], str):
        #         data[i] = data[i].strip().upper()
        #     # standardize the strings data by removing spaces and capitalizing
        self.write_queue.put(data)  # Put the data into the queue
        return 1

    def write_multiple(self, data: list[list]):
        for d in data:
            d = self.standardize_data(d)
            self.write_queue.put(d)  # Put the data into the queue
        return 1
        # self.cursor.executemany(f"INSERT INTO {self.db_name} VALUES(?, ?, ?, ?, ?)", data)
        # self.connection.commit()

    def standardize_data(self, data: list):
        for i in range(len(data)):
            if isinstance(data[i], str):
                data[i] = data[i].strip().upper()
        return data

    # SEARCH FUNCTIONS

    def search_database(self, dict_param: dict):
        for key in dict_param.keys():
            if isinstance(dict_param[key], str):
                dict_param[key] = dict_param[key].strip().upper()
            # standardize the strings data by removing spaces and capitalizing

        if len(dict_param) == 0:
            return self.read_all()

        if len(dict_param) > 3:
            print("too many parameters")
            return -1

        if len(dict_param) == 1:
            if 'gap_name' in dict_param.keys():
                print(f"search by gap name {dict_param['gap_name']} \n")
                return self.search_database_by_name(dict_param['gap_name'])
                # I don't know why but using 'gap_name' as key allows to look for a variable gap_name used as key

            if 'date' in dict_param.keys():
                return self.search_databes_by_date(dict_param['date'])
            
            if 'hour' in dict_param.keys():
                return self.search_database_by_time(dict_param['hour'])

        if len(dict_param) == 2:
            if 'gap_name' in dict_param.keys() and 'date' in dict_param.keys():
                return self.search_database_by_name_and_date(dict_param['gap_name'], dict_param['date'])

            if 'date' in dict_param.keys() and 'hour' in dict_param.keys():
                return self.search_database_by_date_and_time(dict_param['date'], dict_param['hour'])
            
            if 'gap_name' in dict_param.keys() and 'hour' in dict_param.keys():
                return self.search_database_by_name_and_time(dict_param['gap_name'], dict_param['hour'])

        return self.search_database_by_name_date_and_time(dict_param['gap_name'], dict_param['date'], dict_param['hour'])

    def read_all(self):
        # read all data in the table
        self.cursor.execute(f"SELECT * FROM {self.db_name}")
        rows = self.cursor.fetchall()
        if len(rows) == 0:
            print("no data found")
        return rows  # it returns a list of tuples

    def search_database_by_name(self, gap_name):
        self.cursor.execute(f"SELECT * FROM {self.db_name} WHERE gap=?", (gap_name,))
        rows = self.cursor.fetchall()
        if len(rows) == 0:
            print("no data found")
            return 0
        return rows  # it returns a list of tuples

    def search_databes_by_date(self, date):
        self.cursor.execute(f"SELECT * FROM {self.db_name} WHERE date=?", (date,))
        rows = self.cursor.fetchall()
        if len(rows) == 0:
            print("no data found")
            return 0
        return rows  # it returns a list of tuples
    
    def search_database_by_time(self, hour):
        self.cursor.execute(f"SELECT * FROM {self.db_name} WHERE hour=?", (hour,))
        rows = self.cursor.fetchall()
        if len(rows) == 0:
            print("no data found")
            return 0
        return rows  # it returns a list of tuples

    def search_database_by_name_and_date(self, gap_name, date):
        print(f"search by gap name {gap_name} and date {date}\n")
        self.cursor.execute(f"SELECT * FROM {self.db_name} WHERE gap=? AND date=?", (gap_name, date))
        rows = self.cursor.fetchall()
        if len(rows) == 0:
            print("no data found")
            return 0
        return rows  # it returns a list of tuples

    def search_database_by_date_and_time(self, date, hour):
        self.cursor.execute(f"SELECT * FROM {self.db_name} WHERE date=? and hour=?", (date, hour))
        rows = self.cursor.fetchall()
        if len(rows) == 0:
            print("no data found")
            return 0
        return rows  # it returns a list of tuples
    
    def search_database_by_name_and_time(self, gap_name, hour):
        self.cursor.execute(f"SELECT * FROM {self.db_name} WHERE gap=? and hour=?", (gap_name, hour))
        rows = self.cursor.fetchall()
        if len(rows) == 0:
            print("no data found")
            return 0
        return rows  # it returns a list of tuples

    def search_database_by_name_date_and_time(self, gap_name, date, hour):
        self.cursor.execute(f"SELECT * FROM {self.db_name} WHERE gap=? AND date=? AND hour=?", (gap_name, date, hour))
        rows = self.cursor.fetchall()
        if len(rows) == 0:
            print("no data found")
            return 0
        return rows  # it returns a list of tuples

    # DELETE FUNCTIONS

    def delete(self, gaps: list[str]):
        gaps = [gap.strip().upper() for gap in gaps]
        self.cursor.executemany(f"DELETE FROM {self.db_name} WHERE gap=?", (gaps,))
        self.connection.commit()
        return 1

    def delete_all(self):
        self.cursor.execute(f"DELETE FROM {self.db_name}")
        self.connection.commit()

    # CECK INPUT data FUNCTIONS

    def check_input_database(self, data, choise: bool):
        '''return -1 if the input data is not valid, 1 otherwise'''
        if choise:
            return self.check_multiple(data)
        else:
            return self.check_one(data)

    def check_multiple(self, data: list[list]):
        for d in data:
            if self.check_one(d) == -1:
                return -1
        return 1

    def check_one(self, data: list):
        """(gap, location, people_in, people_out, unique_people, date, hour)"""
        if not isinstance(data, list):
            print(f"invalid input data: --{data}--\nplease insert a list")
            return -1

        if len(data) != self.column:
            print(f"invalid input {data}, please insert a list of {self.column} elements, see the interface documentation")
            return -1

        if not isinstance(data[0], str) or len(data[0]) == 0:
            print("invalid input data, please insert a string as gap name")
            return -1

        if not isinstance(data[1], str) or len(data[1]) == 0:
            print("invalid input data, please insert a string as location")
            return -1

        if not isinstance(data[2], int):
            print("invalid input data, please insert an integer as number of people_in")
            return -1
        
        if not isinstance(data[3], int):
            print("invalid input data, please insert an integer as number of people_out")
            return -1
        
        if not isinstance(data[4], int):
            print("invalid input data, please insert an integer as number of unique_people")
            return -1

        if not isinstance(data[5], str) or len(data[5]) == 0:
            print("invalid input data, please insert a string as date")
            return -1

        if not isinstance(data[6], str) or len(data[6]) == 0:
            print("invalid input data, please insert a string as hour")
            return -1

        return 1

class DatabaseSingletonUnity(DatabaseSingleton):
    def __new__(cls, *args, **kwargs):
        if not hasattr(cls, '_singleton_2'):
            # first call
            cls._singleton_2 = object.__new__(cls)
            # already exist        
        return cls._singleton_2
    
    def __init__(self, db_name="database_unity") -> None:
        super().__init__(db_name)

############################################
# INTERFACE CODE

class DbInterface:
    def __init__(self, database: DatabaseSingleton) -> None:
        self.db = database

    def write_db(self, data: list):
        '''
        data = [
            arg 0: "gap name": string
            arg 1: "(latitude, longitude)": string
            arg 2: number of people in: int
            arg 3: number of people out: int
            arg 4: number of unique people: int
            arg 5: date in format "yyyy-mm-dd" : string
            arg 6: hour in format "hh:mm" : string            
            ],

        or

        data = [ (gap, location, people_in, people_out, unique_people, date, hour),
            (gap, location, people_in, people_out, unique_people, date, hour), ...]

        '''
        choise = isinstance(data[0], list)  # if data is a list of list, choise is True, otherwise is False
        print(f"choise = {choise}")
        if self.db.check_input_database(data, choise) == -1:
            return -1
        self.db.write_to_database(data, choise)
        return

    def search_db(self, **kwargs) -> list[tuple]:
        '''
        It reads the data from the db associated to the parameters passed as input.
        - If no parameters are passed, it reads all the db, return a list of tuples, each tuple is a row of the db.
        List of parmaters:
        - gap_name = string
        - date = string
        - hour = string
        If more than one parameter is passed, it returns the rows that satisfy all the parameters.
        Return None if no data is found.
        Return -1 if the list of parameters is not valid.
        '''

        return self.db.search_database(kwargs)

    def delete_db(self, gaps: list[str]):
        '''
        Take as input a list of gap names to delete from the database.
        If "all" is passed as input, it deletes all the database.
        '''
        if (len(gaps) <= 0):
            print("invalid gap")
            return -1
        print(f"deleting {gaps} from the database")
        self.db.delete(gaps)
        if gaps[0] == "all":
            self.db.delete_all()

############################################
# CLIENT CODE

def client_code():
    database = DatabaseSingleton(db_name="general_database")
    database_unity = DatabaseSingletonUnity(db_name="unity_database")

    db = DbInterface(database)
    db_unity = DbInterface(database_unity)

    data_1 = [["varco 1", '(9.23, 31.19)', 15, 18, 10, "2021-09-01", "10:00"],
              ["varco 3", '(9.40, 31.19)', 25, 22, 15, "2021-09-01", "10:00"],
              ["varco 2", '(9.28, 31.19)', 20, 21, 12, "2021-09-01", "10:00"],
              ["varco 2", '(9.28, 31.19)', 30, 30, 30, "2021-09-01", "11:00"],
              ["varco 2", '(9.28, 31.19)', 80, 121, 100, "2021-09-01", "12:00"],
              ["varco 2", '(9.28, 31.19)', 5, 15, 25, "2021-09-02", "10:00"],
            ]
    data_2 = ["varco 2", '(509742.055, 4340858.495)', 51, 24, 20, "2021-09-01", "10:00"]
    db.delete_db(["all"])
    time.sleep(2)
    db.write_db(data = data_1)
    time.sleep(2)
    # db_unity.write_db(data = data_2)
    # db.delete_db(["all"])

    # time.sleep(2)

    # db.write_db(data=data)
    # db.delete_db("all")
    # db.write_db(data=data)
    # rows1 = db.search_db()
    # print(f"ROWS 1: {rows1} \n")
    # print("\n\n")
    rows1 = db.search_db(date = "2021-09-01")
    print(f"ricerca per data: {rows1} \n")

    rows2 = db.search_db(gap_name = "varco 2")
    print(f"ricerca per nome: {rows2} \n")

    rows_2 = db.search_db(hour = "10:00")
    print(f"ricerca per ora: {rows_2} \n")

    rows3 = db.search_db(gap_name = "varco 2", date = "2021-09-01")
    print(f"ricerca per nome e data: {rows3}\n")

    rows4 = db.search_db(date = "2021-09-01", hour = "10:00")
    print(f"ricerca per data e ora: {rows4}\n")
    # db.delete_db(['all'])

    rows5 = db.search_db(gap_name = "varco 2", date = "2021-09-01", hour = "10:00")
    print(f"ricerca per nome, data e ora: {rows5}\n")

    rows6 = db.search_db(gap_name = "varco 2", hour = "11:00") 
    print(f"ricerca per nome e ora: {rows6}\n")

    # for row in rows2:
    #     print(row)
    # print("\n\n")

    # rows3 = db.search_db(gap_name = "varco 2", date = "2021-09-01")
    # print(f"ROWS 3: {rows3}\n")
    # # for row in rows3:
    # #     print(row)
    # print("\n\n")

    # rows4 = db.search_db(date = "2021-09-01", hour = "10:00")
    # print(f"ROWS 4: {rows4}\n")
    # # for row in rows4:
    # #     print(row)
    # print("\n\n")

    # rows5 = db.search_db(date = "2021-09-01")
    # print(f"ROWS 5: {rows5}\n")
    # for row in rows5:
    #     print(row)


if __name__ == "__main__":
    client_code()

