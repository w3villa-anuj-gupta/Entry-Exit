import sqlite3
import sys

# -----------------------------
# Check command-line argument
# -----------------------------
if len(sys.argv) != 2:
    print("Usage: python delete_face.py <name>")
    sys.exit(1)

name = sys.argv[1]
name = name.strip()

#  Sqlite connection 

def delete_name(name):
    conn = sqlite3.connect("faces.db")
    c = conn.cursor()
    c.execute("SELECT name, embedding FROM users")
    rows = c.fetchall()
    for db_name,embeddings in rows:
        if db_name == name:
            c.execute('''
            DELETE FROM users WHERE name = '{}';
            '''.format(name))
            conn.commit()
            return name
    return None


if delete_name(name):
    print(f"{name} is deleted sucessfully")
else:
    print(f'{name} not Exist')