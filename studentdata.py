import psycopg2


# I want to connect to the database to be able to insert and update data in the database 
def connect_db():
    conn=psycopg2.connect(
        dbname='STUDENTS',
        user="postgres",
        password="1234",
        host="localhost",
        port="5432"
    )
    return conn


#Inserting the records into the database 
def insert_student(conn,name,passport_image_path):
    with open(passport_image_path,'rb') as file:
        binary_data=file.read()
        
    cursor=conn.cursor()
    cursor.execute("INSERT INTO student_passports (name, passport_image) VALUES (%s, %s)", (name, binary_data))
    conn.commit()
    cursor.close()
    
    
    
    
def retrieve_students():
    conn = connect_db()
    cur = conn.cursor()

    try:
        # Execute the SELECT query
        cur.execute("SELECT student_id, name, passport_image FROM student_passports")

        # Fetch all the rows
        rows = cur.fetchall()

        # Print the retrieved data (optional)
        for row in rows:
            student_id, name, passport_image = row
            print(f"Student ID: {student_id}, Name: {name}, Passport Image: {passport_image}")
            # Optionally, you can process the passport_image here
            
        return rows
    except psycopg2.Error as e:
        print("Error retrieving student data:", e)
        return None
    finally:
        cur.close()
        conn.close()

# Example usage
students = retrieve_students()
if students is not None:
    print("Retrieved student data successfully!")
    
    

# Example usage
def main():
    conn = connect_db()
    insert_student(conn, "Jeremiah Anku Coblah", "./Jeremiah.jpg")
    conn.close()

if __name__ == "__main__":
    main()
    