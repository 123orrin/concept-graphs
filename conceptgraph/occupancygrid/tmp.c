// 5
void change_int(int* p)
{
    *p = 45;
}
// 10
void change_int_ptr(int** pp)
{
    **pp = 46;
}
// 11
void change_int_storage(int** pp)
{
    *pp = malloc(sizeof(int));
}
// 16
void modify(int* arr)
{
    arr[2] = 8;
}
// 21
typedef struct Student
{
    char name[1000];
    int age;
} Student;
// 27
void change_student_name(Student* p)
{
    strcpy(p->name, "Jenny");
}
// 28
void change_student_age(Student* p)
{
    p->age = 22;
}

int main()
{   
    // 1
    int a = 42;
    // 2
    int* p_a = &a;
    // 3
    *p_a = 43;
    // 4
    int b = 44;
    int* p_b = &b;
    // 6
    change_int(&a);
    // 7 
    change_int(p_a);
    // 8
    int** pp_a;
    // 9
    pp_a = &p_a;
    // 
    //12
    change_int_ptr(pp_a);
    change_int_ptr(&p_a);
    // 13
    change_int_storage(&p_a);
    // 14
    change_int_storage(pp_a);
    // 15
    int arr[] = {5, 6, 7};
    // 17
    modify(arr);
    // 18
    int* p_block = malloc(3 * sizeof(int));
    modify(p_block);
    // 19
    change_int(p_block);
    // 20
    change_int_ptr(&p_block);
    // 21 (con't)
    Student student = {"John Doe", 20};
    // 22
    strcpy(student.name, "Jennifer");
    // 23
    student.age = 21;
    // 24
    Student* p_s = &student;
    // 25
    strcpy(p_s->name, "Jenny");
    // 26
    p_s->age = 20;
    // 29
    change_student_name(p_s);
    // 30 
    change_student_age(&student);
    // 31
    Student s_arr[5];
    // 32
    
}
