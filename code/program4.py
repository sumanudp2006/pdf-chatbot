import sqlite3

schema = """
    CREATE TABLE students(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        city TEXT,
        subject TEXT,
        marks INT
    );
"""

# create sqlite connection
connection = sqlite3.connect("students.db")

# get the data from table
cursor = connection.cursor()

# execute the query
cursor.execute("select id, name, marks, subject from students")

# get the data
students = cursor.fetchall()

# process the data
for student in students:
    print(student)

# close the connection
connection.close()