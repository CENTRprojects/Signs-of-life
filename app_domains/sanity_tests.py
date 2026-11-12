import pkg_resources
from packaging import version
from colorama import Fore, init
import sys
import psycopg2 as pg
from config import RUN_CONFIG

# Initialize colorama
init(autoreset=True)
missing_deps = []
mismatched_deps = []

def check_dependency_version(dependency, required_version, missing_deps, mismatched_deps):
    """
    Check if the installed version of a dependency matches the required version.
    Collect missing and mismatched dependencies.

    :param dependency: The name of the dependency (e.g., 'foobar').
    :param required_version: The required version of the dependency (e.g., '1.2.34').
    :param missing_deps: List to collect missing dependencies.
    :param mismatched_deps: List to collect mismatched dependencies.
    :return: None
    """
    try:
        # Get the installed version of the dependency
        installed_version = pkg_resources.get_distribution(dependency).version
        
        # Compare the installed version with the required version
        if version.parse(installed_version) != version.parse(required_version):
            print(f"{Fore.YELLOW}Alert: The installed version of {dependency} is {installed_version}, "
                  f"but the required version is {required_version}.")
            mismatched_deps.append((dependency, required_version, installed_version))
        else:
            print(f"{Fore.GREEN}The installed version of {dependency} ({installed_version}) matches the required version.")
    
    except pkg_resources.DistributionNotFound:
        print(f"{Fore.RED}Alert: {dependency} is not installed on the system.")
        missing_deps.append((dependency, required_version))
    
    except Exception as e:
        print(f"{Fore.RED}An error occurred while checking {dependency}: {e}")

def check_db_connection():
    """
    Check if a DB connection is possible with the credentials in the config file

    :return: Boolean
    """
    if RUN_CONFIG["USE_DB"] == True:
        print("---------------------------------------------------")
        print("Testing DB connection...")
        try:
            # Establish connection to the database
            connection = pg.connect(
                host=RUN_CONFIG["DBHOST"],
                # port="1234",
                port=RUN_CONFIG["DBPORT"],
                # dbname="brrrrrrt",
                dbname=RUN_CONFIG["DBNAME"],
                user=RUN_CONFIG["DBUSER"],
                password=RUN_CONFIG["DBPASS"]
            )

            # Create a cursor to execute a simple SQL query
            cursor = connection.cursor()
            
            # Execute a simple query (e.g., SELECT version to check DB response)
            cursor.execute("select * from signs_of_life_crawler limit 1")
            db_version = cursor.fetchone()  # Get the result of the query
            print(db_version)

            # close connection
            cursor.close()
            connection.close()
            print("Database connection successful!")
            print("---------------------------------------------------")
            return True
        except pg.OperationalError as e:
            print(f"OperationalError: Could not connect to the database. Details: {e}")
            print("---------------------------------------------------")
            return False
        except pg.DatabaseError as e:
            print(f"DatabaseError: There was an issue with the database operation. Details: {e}")
            print("---------------------------------------------------")
            return False
        except Exception as e:
            print(f"Error connecting to the database: {e}")
            print("---------------------------------------------------")
            return False
    else:
        return True


def display_unmatching_dependencies(missing_deps, mismatched_deps):
    """
    Display a summary of missing and mismatched dependencies 
    
    :param missing_deps: List of missing dependencies.
    :param mismatched_deps: List of mismatched dependencies.
    :return: Boolean
    """
    answer = True
    print (missing_deps)
    if not missing_deps and not mismatched_deps:
        print(f"{Fore.GREEN}All dependencies are correctly installed and up to date.")
        return answer

    print(f"{Fore.YELLOW}\nSummary of issues:")
    if missing_deps:
        answer = False
        print(f"{Fore.RED}Missing dependencies:")
        for dep, required in missing_deps:
            print(f"{Fore.RED}  - {dep}: Required version {required}")
    if mismatched_deps:
        print(f"{Fore.YELLOW}Mismatched dependencies:")
        for dep, required, installed in mismatched_deps:
            print(f"{Fore.YELLOW}  - {dep}: Installed version {installed}, Required version {required}")
            if dep == "numpy":
                answer = False
    
    print(f"\n{Fore.RED}Please review the issues above.")
    install_commands = []

    for dep, required in missing_deps:
        install_commands.append(f"pip install {dep}=={required}")

    for dep, required, installed in mismatched_deps:
        install_commands.append(f"pip install {dep}=={required}")

    print(f"\n{Fore.RED}The following installation commands can be used to fix the issues:")
    for command in install_commands:
        print(f"{Fore.RED}  {command}")
    
    return answer

def run_check():
        # List of dependencies to check
    dependencies_to_check = {
        "numpy": "1.18.3",
        "requests": "2.22.0",
        "joblib": "0.13.0",
        "matplotlib": "3.2.1",
        "pandas": "1.0.3",
        "asyncio": "3.4.3",
        "aiohttp": "3.5.4",
        "beautifulsoup4": "4.6.0",
        "asyncpool": "1.0",
        "cltk": "0.1.117",
        "hazm": "0.7.0",
        "aiodns": "2.0.0",
        "hebrew-tokenizer": "1.0.3",
        "jieba": "0.42.1",
        "konlpy": "0.5.2",
        "langdetect": "1.0.8",
        "openpyxl": "3.0.3",
        "Pillow": "7.1.1",
        "pythainlp": "2.1.4",
        "selenium": "3.141.0",
        "spacy": "2.2.4",
        "tinysegmenter": "0.4",
        "cchardet": "2.1.1",
        "tqdm": "4.45.0",
        "xgboost": "1.0.2",
        "dnspython": "2.3.0",
        "python-Levenshtein": "0.12.1",
        "tldextract": "2.2.2",
        "soynlp": "0.0.493",
        "psycopg2": "2.8.5"
    }

    # Check all dependencies
    for key in dependencies_to_check:
        check_dependency_version(key, dependencies_to_check[key], missing_deps, mismatched_deps)

    # After all checks, print a list of missing/unamtching dependencies
    if not display_unmatching_dependencies(missing_deps, mismatched_deps):
        print("#####################################################################################")
        print("ALERT NOTICE:")
        print("Unmatching dependency version. Terminating run. Please update the python libraries")
        print("#####################################################################################")
        sys.exit("Unmatching dependency version. Terminating run")  # Stop the process
    
    if not check_db_connection():
        print("#####################################################################################")
        print("ALERT NOTICE:")
        print("Database is not connected. The data will not be updated. Terminating process")
        print("#####################################################################################")
        sys.exit("No DB connection")  # Stop the process

if __name__ == '__main__':

    run_check()