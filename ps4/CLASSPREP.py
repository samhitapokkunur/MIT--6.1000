class Employee:

    num_of_emps = 0
    raise_amount = 1.04

    def __init__(self, first, last, pay):
        self.first = first
        self.last = last
        self.pay = pay
        self.email = first + "." + last + "@company.com"

        Employee.num_of_emps += 1 # not using self bc class variable, harder to override

    def fullname(self):
        return '{} {}'.format(self.first,self.last)

    def apply_raise(self):
        self.pay = int(self.pay * self.raist_amoount)


emp_1 = Employee("Corey","Schafer",50000)
emp_2 = Employee("Test","User",60000)

print(Employee.fullname(emp_1))
print(emp_1.fullname())

Employee.raise_amount = 1.05
print(Employee.raise_amount)
emp_1.raise_amount = 1.06
print(emp_1.raise_amount)
print(Employee.raise_amount)
print(emp_1.__dict__)

print(Employee.num_of_emps)
