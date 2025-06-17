import mysql.connector
from mysql.connector import errorcode
from ir_project.config import DB_CONFIG

def setup_database():
    """
    Connects to MySQL, creates the database if it doesn't exist,
    and creates the necessary tables.
    """
    try:
        # Connect to MySQL server
        cnx = mysql.connector.connect(
            host=DB_CONFIG['host'],
            user=DB_CONFIG['user'],
            password=DB_CONFIG['password']
        )
        cursor = cnx.cursor()

        # Create the database if it doesn't exist
        db_name = DB_CONFIG['database']
        cursor.execute(f"CREATE DATABASE IF NOT EXISTS {db_name} DEFAULT CHARACTER SET 'utf8'")
        print(f"Database '{db_name}' created or already exists.")
        cursor.close()
        cnx.close()

        # Connect to the specific database
        cnx_db = mysql.connector.connect(**DB_CONFIG)
        cursor_db = cnx_db.cursor()

        # Create documents table
        table_name = 'documents'
        table_description = (
            f"CREATE TABLE IF NOT EXISTS `{table_name}` ("
            "  `id` int(11) NOT NULL AUTO_INCREMENT,"
            "  `doc_id` varchar(255) NOT NULL,"
            "  `text` LONGTEXT NOT NULL,"
            "  `metadata` JSON DEFAULT NULL,"
            "  PRIMARY KEY (`id`),"
            "  UNIQUE KEY `doc_id` (`doc_id`)"
            ") ENGINE=InnoDB"
        )
        print(f"Creating table `{table_name}`... ", end='')
        cursor_db.execute(table_description)
        print("Done.")

        cursor_db.close()
        cnx_db.close()

    except mysql.connector.Error as err:
        if err.errno == errorcode.ER_ACCESS_DENIED_ERROR:
            print("Something is wrong with your user name or password")
        elif err.errno == errorcode.ER_BAD_DB_ERROR:
            print("Database does not exist")
        else:
            print(err)
    else:
        print("Database setup completed successfully.")

if __name__ == '__main__':
    setup_database()
