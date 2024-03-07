import psycopg2
import numpy as np

# Function to connect to the database
def connect_db():
    conn = psycopg2.connect(
        dbname='STUDENTS',
        user='postgres',
        password='1234',
        host='localhost',
        port='5432'
    )
    return conn

# Function to insert records into the database
def insert_student(conn, name, passport_image_path, db_face_encoding):
    # Serialize the numpy array to bytes
    db_face_encoding_bytes = db_face_encoding.tobytes()
    
    with open(passport_image_path, 'rb') as file:
        binary_data = file.read()
        
    cursor = conn.cursor()
    cursor.execute("INSERT INTO student_passports (name, passport_image, face_encoding) VALUES (%s, %s, %s)",
                   (name, binary_data, db_face_encoding_bytes))
    conn.commit()
    cursor.close()

# Function to retrieve records from the database
def retrieve_students():
    conn = connect_db()
    cur = conn.cursor()

    try:
        # Execute the SELECT query
        cur.execute("SELECT student_id, name, passport_image, face_encoding FROM student_passports")

        # Fetch all the rows
        rows = cur.fetchall()

        # Print the retrieved data (optional)
        for row in rows:
            student_id, name, passport_image, face_encoding_bytes = row
            db_face_encoding = np.frombuffer(face_encoding_bytes, dtype=np.float64)
            db_face_encoding = db_face_encoding.reshape((128,))
            print(f"Student ID: {student_id}, Name: {name}, Passport Image: {passport_image}, Face Encoding: {db_face_encoding}")
            # Optionally, you can process the passport_image here
            
        return rows
    except psycopg2.Error as e:
        print("Error retrieving student data:", e)
        return None
    finally:
        cur.close()
        conn.close()

# Function to alter the table and add the new column
def add_face_encoding_column():
    conn = connect_db()
    cur = conn.cursor()

    try:
        # Alter the table to add the new column
        cur.execute("ALTER TABLE student_passports ADD COLUMN face_encoding BYTEA;")
        conn.commit()
        print("Added 'face_encoding' column successfully!")
    except psycopg2.Error as e:
        print("Error adding 'face_encoding' column:", e)
    finally:
        cur.close()
        conn.close()

# Example usage
def main():
    #add_face_encoding_column()
    conn = connect_db()
    face_encoding = np.random.rand(128)  # Example numpy array for face encoding
    insert_student(conn, "Elon Musk", "./elon musk.jpg", face_encoding)
    conn.close()

if __name__ == "__main__":
    main()
