import numpy as np


class DES():
    

    p1 = np.array([[57,49,41,33,25,17,9],
                    [1,58,50,42,34,26,18],
                    [10,2,59,51,43,35,27],
                    [19,11,3,60,52,44,36],
                    [63,55,47,39,31,23,15],
                    [7,62,54,46,38,30,22],
                    [14,6,61,53,45,37,29],
                    [21,13,5,28,20,12,4]],dtype="int")

    p2 = np.array([[14,17,11,24,1,5],
                    [3,28,15,6,21,10],
                    [23,19,12,4,26,8],
                    [16,7,27,20,13,2],
                    [41,52,31,37,47,55],
                    [30,40,51,45,33,48],
                    [44,49,39,56,34,53],
                    [46,42,50,36,29,32]], dtype="int")


    p = np.array([[16,7,20,21],
                [29,12,28,17],
                [1,15,23,26],
                [5,18,31,10],
                [2,8,24,14],
                [32,27,3,9],
                [19,13,30,6],
                [22,11,4,25]],dtype="int")

    S_Box=np.array([
                [[14,4,13,1,2,15,11,8,3,10,6,12,5,9,0,7],
                    [0,15,7,4,14,2,13,1,10,6,12,11,9,5,3,8],
                    [4,1,14,8,13,6,2,11,15,12,9,7,3,10,5,0],
                    [15,12,8,2,4,9,1,7,5,11,3,14,10,0,6,13]],
                    
                [[15,1,8,14,6,11,3,4,9,7,2,13,12,0,5,10],
                [3,13,4,7,15,2,8,14,12,0,1,10,6,9,11,5],
                [0,14,7,11,10,4,13,1,5,8,12,6,9,3,2,15],
                [13,8,10,1,3,15,4,2,11,6,7,12,0,5,14,9]],
                
                [[10,0,9,14,6,3,15,5,1,13,12,7,11,4,2,8],
                [13,7,0,9,3,4,6,10,2,8,5,14,12,11,15,1],
                [13,6,4,9,8,15,3,0,11,1,2,12,5,10,14,7],
                [1,10,13,0,6,9,8,7,4,15,14,3,11,5,2,12]],
                
                [[7,13,14,3,0,6,9,10,1,2,8,5,11,12,4,15],
                [13,8,11,5,6,15,0,3,4,7,2,12,1,10,14,9],
                [10,6,9,0,12,11,7,13,15,1,3,14,5,2,8,4],
                [3,15,0,6,10,1,13,8,9,4,5,11,12,7,2,14]],
                
                [[2,12,4,1,7,10,11,6,8,5,3,15,13,0,14,9],
                [14,11,2,12,4,7,13,1,5,0,15,10,3,9,8,6],
                [4,2,1,11,10,13,7,8,15,9,12,5,6,3,0,14],
                [11,8,12,7,1,14,2,13,6,15,0,9,10,4,5,3]],
                
                [[12,1,10,15,9,2,6,8,0,13,3,4,14,7,5,11],
                [10,15,4,2,7,12,9,5,6,1,13,14,0,11,3,8],
                [9,14,15,5,2,8,12,3,7,0,4,10,1,13,11,6],
                [4,3,2,12,9,5,15,10,11,14,1,7,6,0,8,13]],
                
                [[4,11,2,14,15,0,8,13,3,12,9,7,5,10,6,1],
                [13,0,11,7,4,9,1,10,14,3,5,12,2,15,8,6],
                [1,4,11,13,12,3,7,14,10,15,6,8,0,5,9,2],
                [6,11,13,8,1,4,10,7,9,5,0,15,14,2,3,12]],
                
                [[13,2,8,4,6,15,11,1,10,9,3,14,5,0,12,7],
                [1,15,13,8,10,3,7,4,12,5,6,11,0,14,9,2],
                [7,11,4,1,9,12,14,2,0,6,10,13,15,3,5,8],
                [2,1,14,7,4,10,8,13,15,12,9,0,3,5,6,11]]
                
                ],dtype="int")


    def permutaion_choice1(self,key):
        #takes the key as vector (if matrix we will convert it) not matrix and returns matrix
        key= key.reshape((1,-1))[0]
        matrix= np.zeros_like(self.p1,dtype="int")
        for r in range(len(self.p1)):
            for c in range(len(self.p1[r])):
                matrix[r,c]=key[self.p1[r,c]-1]
        
        return matrix #(8,7)


    def shift(self,sub_key):
        m= np.zeros_like(sub_key,dtype="int")
        f=sub_key[:,0]
        m[:,0:6]= sub_key[:,1:]
        m[:,-1]=f[::-1]
        return m



    def shift_left(self,sub_key,amount):
        for _ in range(amount):
            sub_key=self.shift(sub_key)
        
        return sub_key

    def new_shift(self,C,D,amount):
        for _ in range(amount):
            CN=np.zeros_like(C,dtype="int")
            DN=np.zeros_like(D,dtype="int")
            c0=C[0]
            CN[:-1]=C[1:]
            CN[-1]=c0

            d0=D[0]
            DN[:-1]=D[1:]
            DN[-1]=d0
        
        return CN,DN

    def left_circular_shift(self,key, amount):
        #takes the key as matrix
        # matrix= np.zeros_like(key,dtype="int")

        up = key[:4]
        down=key[4:]
        C=up.reshape((1,-1))[0]
        D=down.reshape((1,-1))[0]
        

        n_C,n_D= self.new_shift(C,D,amount)
        return n_C,n_D

        # matrix[:4]=self.shift_left(up,amount)
        # matrix[4:]=self.shift_left(down,amount)


        # return matrix #(8,7)



    def permutaion_choice2(self,key):
        #takes the key as vector (if matrix we will convert it) not matrix and returns matrix
        key=key.reshape((1,-1))[0]
        matrix= np.zeros_like(self.p2,dtype="int")
        for r in range(len(self.p2)):
            for c in range(len(self.p2[r])):
                matrix[r,c]=key[self.p2[r,c]-1]
        
        return matrix #(8,6)


    def initial_permutation(self,plain):
        #takes plain text as vector (if matrix we will convert it) and returns matrix 
        # plain=plain.reshape((1,-1))[0]
        # plain = plain.reshape((8,8))
        matrix = np.zeros((8,8),dtype="int")

        l=[2,4,6,8,1,3,5,7]
        
        i=0
        for col in l:
            matrix[i]= plain[:,col-1][::-1]
            i+=1
        
        return matrix #(8,8)

    def initial_permutation_inverse(self,cipher):
        #takes cipher text as vector (if matrix we will convert it) and returns matrix 
        # cipher=cipher.reshape((1,-1))[0]
        # cipher = cipher.reshape((8,8))
        matrix = np.zeros((8,8),dtype="int")

        l=[2,4,6,8,1,3,5,7]
        
        i=0
        for col in l:
            matrix[:,col-1]= cipher[i][::-1]
            i+=1
        
        return matrix #(8,8)

        





    def expansion(self,R):
        #takes matrix 4*8 (or 8*4) and return 8*6 matrix

        matrix = np.zeros((8,6),dtype="int")
        R=R.reshape((8,4))

        matrix[:,1:5]=R

        right= R[:,3]
        left= R[:,0]

        matrix[1:,0]=right[0:-1]
        matrix[0,0]=right[-1]

        matrix[:-1,-1]=left[1:]
        matrix[-1,-1]= left[0]

        return matrix

    def sbox(self,R_expanded):
        #takes 8*6 matrix and returns 8*4 matrix
        matrix = np.zeros((8,4),dtype="int")
        for i in range(8):
            s=self.S_Box[i]
            row= R_expanded[i]
            s_r=int(str(row[0])+str(row[-1]) ,2)
            s_col = int(str(row[1])+str(row[2])+str(row[3])+str(row[4]) ,2)
            v= bin(s[s_r,s_col])[2:]
            v=[int(i) for i in v]
            if len(v)<4:
                    for _ in range(4-len(v)):
                        v.insert(0,0)
            v=np.array(v,dtype="int")
            matrix[i]=v
        
        return matrix

    def permutaion(self,R):
        #takes vector (if matrix we will convert it) not matrix and returns matrix
        R=R.reshape((1,-1))[0]
        matrix= np.zeros_like(self.p,dtype="int")
        for r in range(len(self.p)):
            for c in range(len(self.p[r])):
                matrix[r,c]=R[self.p[r,c]-1]
        
        return matrix #(8,4)



    def F(self,R,key):
        #takes matrix R and matrix key(8,6) and return (8,4) matrix

        exp_R= self.expansion(R) #(8,6) matrix
        xor1= np.bitwise_xor(exp_R,key) #(8,6) matrix
        sub=self.sbox(xor1) #(8,4) matrix
        per=self.permutaion(sub) #(8,4) matrix
        return per


    def round(self,L,R,K):
        #takes L(4,8) R(4,8) K(8,6)

        f_out =self.F(R,K) #(8,4) matrix
        f_out=f_out.reshape((4,8)) #(4,8) matrix like L

        new_R= np.bitwise_xor(L,f_out)
        new_L=R

        return new_L , new_R


    def swap(self,L,R):
        return R, L


    def generate_key(self,Key):
        #takes (8,8) key and return (16,8,6) 16 keys one for each round
        OUT_56= self.permutaion_choice1(Key) #PC1 Output
        Keys= np.zeros((16,8,6),dtype="int")
        amounts=[1,1,2,2,2,2,2,2,1,2,2,2,2,2,2,1]
        for i in range(16):
            C,D=self.left_circular_shift(OUT_56,amounts[i])
            OUT_56[:4]=C.reshape((4,7))
            OUT_56[4:]=D.reshape((4,7))

            Keys[i]= self.permutaion_choice2(OUT_56) #PC2 Output
        
        return Keys





    def encrypt(self,plain,Key):
        #plain is (8,8) and key is (8,8) and return (8,8) cipher

        IP= self.initial_permutation(plain) #(8,8)
        L= IP[:4]
        R= IP[4:]

        Keys=self.generate_key(Key)

        for i in range(16):
            L,R= self.round(L,R,Keys[i])
        
        L,R= self.swap(L,R) #Round 17 Swapping
        cipher= np.zeros((8,8),dtype="int")
        cipher[:4]=L
        cipher[4:]=R

        cipher = self.initial_permutation_inverse(cipher)
        return cipher
    

    def decrypt(self,cipher,Key):
        IP= self.initial_permutation(cipher) #(8,8)
        L= IP[:4]
        R= IP[4:]

        Keys=self.generate_key(Key)

        for i in range(16):
            L,R= self.round(L,R,Keys[-(i+1)])
        
        L,R= self.swap(L,R) #Round 17 Swapping
        plain= np.zeros((8,8),dtype="int")
        plain[:4]=L
        plain[4:]=R

        plain = self.initial_permutation_inverse(plain)
        return plain

            

def prepare_text(text):
    #takes text of hexa string of 16 chars and converts it to (8,8) matrix of binary
    matrix = np.zeros((8,8),dtype="int")


    c=""
    i=0
    for char in text:
        c+=char
        if len(c)==2:
            ce=bin(int(c,16))[2:]
            ce=[int(i) for i in ce]
            if len(ce)<8:
                    for _ in range(8-len(ce)):
                        ce.insert(0,0)
            ce=np.array(ce,dtype="int")
            matrix[i]=ce
            i+=1
            c=""
    return matrix

def return_text(text):
    #takes (8,8) matrix and returns it back to hexa string of 16 chars
    c=""
    for w in text:
        w=w.astype("str")
        c+=hex(int("".join(w[:4]),2))[2:]
        c+=hex(int("".join(w[4:]),2))[2:]
    return c.upper()


            
        

        





if __name__ == "__main__":

    plain=""
    key=""
    while len(plain)!=16:
        plain = input("Enter Text: ")
        if len(plain)!=16:
            print("Error, Text must be 16 character Hexadecimel!")
    
    while len(key)!=16:
        key = input("Enter Key: ")
        if len(key)!=16:
            print("Error, Key must be 16 character Hexadecimel!")
    
    num_times=int(input("Enter number of times to run the algo: "))
    mode= int(input("Enter Mode (1 for encryption and 0 for decryption): "))


    plain = prepare_text(plain)
    key= prepare_text(key)

    des=DES()
    for k in range(num_times):
        if mode:
            plain = des.encrypt(plain,key)
            if k == num_times-1:
                plain=np.array(np.logical_not(plain),dtype="int" )
        else:
            if k==0:
                plain=np.array(np.logical_not(plain),dtype="int" )
            plain = des.decrypt(plain,key)

    
    print("Cipher: ")
    print(return_text(plain))


    