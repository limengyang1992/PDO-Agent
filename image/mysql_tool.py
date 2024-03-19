import os
import json
import glob
import pymysql

class MySQLJSONStorage:
    def __init__(self, host='39.99.241.32', user='root', password='18952024', database='PDO'):
        self.host = host
        self.user = user
        self.password = password
        self.database = database
        self.connection = pymysql.connect(
            host=host,
            user=user,
            password=password,
            database=database,
            charset='utf8mb4',
            cursorclass=pymysql.cursors.DictCursor
        )

    # 判断是否连接成功，如果成功返回True，否则重新连接
    def connected(self):
        if self.connection.ping(reconnect=True):
            print('数据库连接成功')
        else:
            self.connection = pymysql.connect(
                host=self.host,
                user=self.user,
                password=self.password,
                database=self.database,
                charset='utf8mb4',
                cursorclass=pymysql.cursors.DictCursor
            )
            print('数据库重新连接成功')

    def create_table(self):
        self.connected()
        with self.connection.cursor() as cursor:
            create_table_query = '''
            CREATE TABLE IF NOT EXISTS json_storage (
                id INT AUTO_INCREMENT PRIMARY KEY,
                dir VARCHAR(255),
                name VARCHAR(255),
                type VARCHAR(50),
                json_data JSON,
                status VARCHAR(50)
            )
            '''
            cursor.execute(create_table_query)
        self.connection.commit()
        self.connection.close()

    def delete_table(self):
        self.connected()
        with self.connection.cursor() as cursor:
            delete_table_query = '''
            DROP TABLE json_storage
            '''
            cursor.execute(delete_table_query)
        self.connection.commit()
        self.connection.close()


    def insert_json_files(self, folder_path,types='img'):
        self.connected()
        with self.connection.cursor() as cursor:
            for i,filename in enumerate(glob.glob(os.path.join(folder_path, '*/*.json'))):
                print(i,filename)
                with open(filename, 'r') as file:
                    data = json.load(file)
                    dir_name = os.path.split(os.path.dirname(filename))[-1]
                    file_name = os.path.basename(filename).split('.')[0]
                    json_data = json.dumps(data)
                    status = '0'  # or whatever initial status you prefer
                    insert_query = '''
                    INSERT INTO json_storage (dir, name, type, json_data, status) 
                    VALUES (%s, %s, %s, %s, %s)
                    '''
                    cursor.execute(insert_query, (dir_name, file_name, types, json_data, status))
        self.connection.commit()
        self.connection.close()

    # 计算json文件的数量
    def count_json_files(self):
        self.connected()
        with self.connection.cursor() as cursor:
            count_query = '''
            SELECT COUNT(*) FROM json_storage
            '''
            cursor.execute(count_query)
            result = cursor.fetchone()
            return result['COUNT(*)']

    # 计算status=0和1的json文件的数量  
    def count_pending_json_files(self):
        self.connected()
        with self.connection.cursor() as cursor:
            count_query = '''
            SELECT COUNT(*) FROM json_storage WHERE status = '0'
            '''
            cursor.execute(count_query)
            result = cursor.fetchone()
            return result['COUNT(*)']


    # 获取其中一条status=0的json文件
    def get_pending_json_file(self):
        self.connected()
        with self.connection.cursor() as cursor:
            select_query = '''
            SELECT * FROM json_storage WHERE status = '0' LIMIT 1
            '''
            cursor.execute(select_query)
            result = cursor.fetchone()
            return result
    # 根据name更新status
    def update_status(self, name, status):
        self.connected()
        with self.connection.cursor() as cursor:
            update_query = '''
            UPDATE json_storage SET status = %s WHERE name = %s
            '''
            cursor.execute(update_query, (status, name))
        self.connection.commit()
        self.connection.close()

# Example usage:
if __name__ == "__main__":
    db = MySQLJSONStorage()
    # db.delete_table()
    # db.create_table()
    # db.insert_json_files('/home/mengyang/dataset/config_img/')
    print(db.count_json_files())
    print(db.count_pending_json_files())
    res = db.get_pending_json_file()
    print(res)
    name = res['name']
    ids = res['id']
    print(name, ids)
    # db.update_status(name)

