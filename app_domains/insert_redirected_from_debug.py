from config import RUN_CONFIG
import psycopg2 as pg
import sys
from os import listdir
from os.path import isfile, join, dirname, abspath, basename
import csv
import shutil
from datetime import date
# performance logger
from utils import PerformanceLogger
plog = PerformanceLogger(filepath=join(RUN_CONFIG["MAIN_DIR"],"logging"), filename="db_redirect_perf.log", enable_logging = RUN_CONFIG['PERFORMANCE_LOGGING'])


DB_FIELDS = set()
db = None
input_urls = set()

def executeSQLFromFile(filename):
    with open(filename, 'r') as f:
        sqlCommands = f.read().replace('\n', '').split(";")
    cur = db.cursor()
    for command in sqlCommands:
        if len(command) > 0:
            try:
                cur.execute(command)
            except pg.ProgrammingError:
                db.rollback()
                raise
    cur.close()
    db.commit()
def ConnectDB():
    print("\nConnecting to DB...")
    global db
    try: 
        # first check the db exists
        db = pg.connect(host=RUN_CONFIG["DBHOST"], port=RUN_CONFIG["DBPORT"], database=RUN_CONFIG["DBNAME"], user=RUN_CONFIG["DBUSER"], password=RUN_CONFIG["DBPASS"])
        cur = db.cursor()
        # then check the table exists
        try:
            cur.execute("select * from signs_of_life_crawler limit 1")
        except (Exception, pg.ProgrammingError) as error:
            # if we're here it means we connected to the db but there is no table
            plog.it(f"Error connecting to table: {error}", is_error=True)
            db.rollback()
            sqlfile = join(dirname(abspath(__file__)), "sql", "setup_crawler_db_1.sql")
            executeSQLFromFile(sqlfile)
            sqlfile = join(dirname(abspath(__file__)), "sql", "setup_crawler_db_2.sql")
            executeSQLFromFile(sqlfile)
            sqlfile = join(dirname(abspath(__file__)), "sql", "setup_crawler_db_3.sql")
            executeSQLFromFile(sqlfile)
            sqlfile = join(dirname(abspath(__file__)), "sql", "setup_crawler_db_4.sql")
            executeSQLFromFile(sqlfile)
            sqlfile = join(dirname(abspath(__file__)), "sql", "setup_crawler_db_5.sql")
            executeSQLFromFile(sqlfile)
            cur.execute("select * from signs_of_life_crawler limit 1")
        q = cur.fetchall()
        cur.close()
        print("DB Connection successful.")
    except (Exception, pg.Error) as error:
        plog.it(f"Error while connecting to PostgreSQL: {error}", is_error=True)
        raise

def DiscoverFiles():
    print("\nLooking for output files...")
    csvpath = dirname(RUN_CONFIG["output_final_file_path"])
    plog.it("CSVPath: " + csvpath)
    outputfiles=[join(csvpath,f) for f in listdir(csvpath) if isfile(join(csvpath, f))]
    print("Found {} output files.".format(len(outputfiles)))
    return outputfiles

def ProcessFiles(outputfiles):
    print("\nScanning files...")
    for file in outputfiles:
        print("Checking {}".format(file))
        if "final_" in file and "_DEBUG" in file:
            print("\n\tPROCESSING FILE: {}".format(file))
            ProcessFile(file)
    print("Finished scanning all files.")

def ConvertBool(csvbit):
    """csv is sending in various values for bool (eg 1.0, "1.0", True and so on), but python is converting everything to true and null to false.
    Have to parse it properly before handing it over to psycopg2 to insert into postgres"""
    if csvbit == "":
        return "null"
    else:
        return csvbit
    #elif csvbit in [1.0,1,True,"1.0","1","true","True"]:
    #    return True
    #elif csvbit in [0.0,0,False,"0.0","0","false","False"]:
    #    return False
    #elif type(csvbit) == type(''):
    #    return csvbit
    raise ValueError('Unknown boolean type: {}({})'.format(csvbit, type(csvbit)))
def FilterColumns():
    """We're only updating specific columns, but we need to match the right files
    """
    global DB_FIELDS
    #TABLEAU_COLUMNS = ["is_redirected","is_redirected_different_domain", "redirection_type"]
    TABLEAU_COLUMNS = ["all_status_codes", "comment"]
    for col in TABLEAU_COLUMNS:
        DB_FIELDS.add(col)
    plog.it("DB_FIELDS: {}".format(DB_FIELDS))

def RetrieveDBFilename(new_filename):
    """Find the closest matching filename in the database, which should have the same 
    fingerprint as the one we're trying to update from
    """
    cur = db.cursor()
    fingerprint = new_filename.strip('res_').replace('_DEBUG','')
    plog.it(f"Attempting to match {new_filename} using fingerprint: {fingerprint}")
    q = f"select filename from signs_of_life_crawler where filename like '%{fingerprint}' limit 1"
    cur.execute(q)
    r = cur.fetchone()
    plog.it(f"\tReceived: {r}")
    return r[0] if r else None

def ConstructSQLBase():
    """
        Get the columns from formatting.py and use that to construct the insert
    """
    global DB_FIELDS
    msg = "update signs_of_life_crawler as solc set "
    for col in DB_FIELDS:
        msg += col.lower() + "=solc2." + col.lower() + ", "
    msg = msg[:-2] + " from (values "
    return msg

def ConstructSQLData(csvfile, db_filename, row):
    """
        Extract the data from the csv row (dict) and construct a string to be added to the qstr
    """
    global DB_FIELDS, input_urls
    try:
        msg = "($_${}$_$,$_${}$_$,".format(db_filename,row['input_url'])
    except:
        col_names = '\n'.join(row.keys())
        plog.it(f"ERROR ON CSV : {csvfile} | db_filename : {db_filename} | type(row) : {type(row)} | row columns : {col_names}")
        raise
    for col in DB_FIELDS:
        try:
            msg += "$_${}$_$,".format(ConvertBool(row[col]))
        except KeyError:
            msg += "$_${}$_$,".format('null')
            
    msg = msg[:-1] # get rid of extra comma
    msg += "), "
    return msg

def ConstructSQLTail(qstr):
    """
        Get the columns from DB_FIELDS and construct the on conflict rules
    """
    # msg_tail = " on conflict (filename,date,input_url) do update set filename = excluded.filename, "
    #for col in DB_FIELDS:
    #    msg_tail += f"{col} = excluded.{col}, "
    #msg_tail = msg_tail[:-2]
    # msg_tail = " on conflict (filename,date,input_url) do nothing"
    msg_tail = ") as solc2(filename,input_url,"
    for col in DB_FIELDS:
        msg_tail += f"{col.lower()},"
    msg_tail = msg_tail[:-1]+") where solc2.filename = solc.filename and solc2.input_url = solc.input_url"
    return qstr[:-2]+msg_tail

def ProcessFile(file):
    # get a good filename, check if we're trying to input this data again (due to some error)
    base_filename = basename(file)
    if "-DB_" not in file:
        filename = date.today().strftime("%Y-%m-%d")+"-DB_"+base_filename
    else:
        filename = base_filename
    plog.it("filename: " + filename)
    destination = join(dirname(dirname(abspath(__file__))), "completed", filename)
    plog.it("destination: "+ destination)
    
    # Setup the query string that we'll use to input into the database
    qlist = []
    # we also need to make sure we're targeting the correct input_url in the existing database
    new_filename = base_filename
    db_filename = RetrieveDBFilename(new_filename)
    if not db_filename:
        plog.it(f"ahhh! I can't find match this file to anything in the database: {new_filename}")
        return
    qbase = ConstructSQLBase()
    qstr = qbase
    # Open file, read it, process it.
    plog.it("input file: "+ file)
    plog.it("qstr: {}".format(qstr))
    with open(file, newline='', encoding="utf-8-sig") as csvfile:
        csv_delimiter = ','
        header = csvfile.readlines()[0]
        if '|' in header:
            csv_delimiter = '|'
    with open(file, newline='', encoding="utf-8-sig") as csvfile:
        reader = csv.DictReader(csvfile, delimiter=csv_delimiter)
        for row in reader:
            #plog.it(row)
            if len(qstr) > 10000:
                qlist.append(ConstructSQLTail(qstr))
                qstr = qbase
            # add to the query_string for this row            
            # check domain isn't nan

            qstr += ConstructSQLData(csvfile, db_filename, row)
        qlist.append(ConstructSQLTail(qstr))
    # check we're not duplicating the tlds we found with the filename
    #CheckDB(filename)
    # when all looks good, submit it to the database
    plog.it("qstr: "+ qstr)
    plog.it("qlist len: {}".format(len(qlist)))
    WriteToDB(qlist)
    # then move the file to the completed folder
    try:    
        shutil.move(file, destination)
    except (Exception) as error:
        print(error)
        input("Close the csv if it is open and press a key to try again.")
        try:
            shutil.move(file, destination)
        except (Exception) as error1:
            plog.it(str(error1.msg), is_error=True)
            plog.it("Critical failure...I'm giving up", is_error=True)

def WriteToDB(qlist):
    print("\tWriting to database")
    plog.it("There are {} qstr in qlist".format(len(qlist)))
    for qstr in qlist:
        try:
            cur = db.cursor()
            cur.execute(qstr)
            db.commit()
            cur.close()
        except (Exception, pg.DatabaseError) as error:
            plog.it(error, is_error=True)
            plog.it(f"Error string: {qstr}", is_error=True)
            db.rollback()
            raise
    print("\tDatabase written")

def DoIt(filename=None):
    plog.perf_go("Starting output processing")
    if filename is None:
        outputFiles = DiscoverFiles()
    else:
        outputFiles = [filename]
    if len(outputFiles) > 0:
        plog.it("Connecting to DB.")
        ConnectDB()
        plog.it("Filtering Columns.")
        FilterColumns()
        plog.it("Processing Files")
        ProcessFiles(outputFiles)
        plog.it("Closing DB connection")
        if(db):
            db.close()
    else:
        plog.it("\nNo output files found. Have you run the rest of the program?")
    plog.perf_end("Completed output processing")

if __name__ == '__main__':
    if len(sys.argv) > 1:
        DoIt(sys.argv[1])
    else:
        DoIt()
