---
title: Quantum Cryptography - Voting Scheme
linkTitle: 11 - Quantum Cryptography
weight: 11
---

##  What we need to know about Quantum Cryptography

## Back in Time  - The beginnings

Cyptography is the science, at the crossroads of mathematics, physics, and computer science, that tends to design protocols to prevent malicious third-party from reading private messages. Even if the development of computers during the 20th century made the research in cryptography explode, the use of cryptographic methods was common before. It is believed that Julius CAESAR used an encryption method, today known as **Caesar Cipher** or **Alphabet Shift Cipher**. The principle is very simple: imagine you want to encode the 26 letters of the Roman alphabet (A to Z), you then assign each letter a number (A is 0, B is 1, ..., Z is 25). You then choose a secret key which is a non-zero integer. The encrypted message is composed of letters, the code of each letter being given by the modulo 26 addition between the code of the original letter and the secret key. Basically, the alphabet is shifted by a constant:


<span style="color:red">**Like PASQAL with s=3 will be SDVTDO**</span>


``` python
# A Python program to illustrate Caesar Cipher Technique
def encrypt(text, s):
    result = ""

    # Traverse the text
    for i in range(len(text)):
        char = text[i]

        # Encrypt uppercase characters
        if char.isupper():
            result += chr((ord(char) + s - 65) % 26 + 65)

        # Encrypt lowercase characters
        else:
            result += chr((ord(char) + s - 97) % 26 + 97)

    return result

# Check the above function
text = "PASQAL"  # INPUT 
s = 3   
print("Text: " + text)
print("Shift: " + str(s))
print("Cipher: " + encrypt(text, s))
```

Results: 

Text: PASQAL
Shift: 3
Cipher: SDVTDO



If we note $n$ the position of the char in the string, $U_n$ the unencrypted char, En the encrypted char, and $K$ the shift:
$$E_{n}=U_{n}+K\text{mod} \left[ 26\right]$$  

<span style="color:red">For instance,</span>


This table displays the positions of each character in the message "PASQAL," the unencrypted characters, the result of $U_n +K$ the result of 
$U_{n}+K\text{mod} \left[ 26\right]$, and the encrypted characters using a Caesar Cipher with a secret key of 3, with the appropriate formatting for mathematical expressions.

| n | Char | Unencrypted Char | \(U_{n}\) | \(U_{n} + K\) | \(U_{n} + K \mod 26\) | Encrypted Char |
|---|------|------------------|-----------|--------------|-----------------------|----------------|
| 1 |  P   |       15         |     15    |      18      |           18          |        R       |
| 2 |  A   |        0         |      0    |       3      |            3          |        D       |
| 3 |  S   |       18         |     18    |      21      |           21          |        U       |
| 4 |  Q   |       16         |     16    |      19      |           19          |        T       |
| 5 |  A   |        0         |      0    |       3      |            3          |        D       |
| 6 |  L   |       11         |     11    |      14      |           14          |        O       |


This code is very simple to break, as you can easily find patterns. For instance, you can compare the frequencies of each letter of the alphabet. In the English language, the most common letter is E with a frequency of 12.7%, followed by the letter T with a frequency of 9.06% . You then compute the frequencies of the encrypted message. The most common letter in the encrypted text will probably be the encrypted char corresponding to E, or T. This method can be applied to all encryption schemes were a letter is always transformed to the same letter.

For this scheme, it also just possible to test the 25 possible secret keys and stop when you have a meaningful message.

Fortunately, cryptology has evolved since them. Even before the invention of the computer, the Germans used a very secure scheme (at that time) to encrypt message during the Second World War. It was secure in two ways:
 - The setup has high combinatorial complexity. Germans have to choose 3 rotors in 5 possibil- ities, choose each initial positions for the rotors (26 possibilities each), and then choose 10 pairs in the plugboard, resulting in the combinatorial complexity

 $$5\times 4\times 3\times 26^{3}\times \frac{26!}{6!\times 10!\times 2^{10}} =158,962,555,217,826,360,000$$

 This is not possible to test all possibilities for a human, nor at that time for a machine.

- Unlike the Caesar Cipher, a letter would not become the same letter every time, as the ro- tors were moving at each letter. This is important because pattern methods (to compare frequencies of the language with the ones in the encrypted text) cannot be used.

The Enigma machine was broken by a team mainly lead by Alan Turing using a machine called Bombe and some flaws of the Enigma machine. In the end, the Enigma configuration could be found in 20 minutes.

## Up-to-date Cryptography

 With the arrival of computers and communication networks came also a greater need for cryptography.

In our modern world, cryptography is ubiquitous. When you load a web page in a browser with a small padlock next to the URL, you are using symmetric and asymmetric encryption without knowing it. When you open some messaging app, like WhatsApp, you are using, again without knowing it (I mean without the user making anything for the cryptographic exchange to happen. It is totally transparent for the user) symmetric and asymmetric encryption, providing end-to-end encryption.

Symmetric and asymmetric encryption are different in their essence. A symmetric encryption scheme uses a secret key to encrypt and to decrypt the message. This means that the two (or more) parties that are communicating need to have the same, and secret, key. Symmetric schemes have several issues:
- The key exchange must be truly secret to ensure the privacy of the encrypted message;
- If you use the key to encrypt more than one time, information can be extracted from the encrypted message;
- The protocol doesn't scale well (i.e. if you want to have encrypted discussion between several users, you have to share the secret key between all the users, which mean, more chance of information leakage).


They have also some advantages: there are easier to understand and to implement and they need less computational power than asymmetric encryption.

In contrast, an asymmetric encryption scheme or public key encryption scheme is a scheme where two keys are used:

- The **Public key** is used only to encrypt messages. It can be distributed without risk to anyone who wants to send an encrypted message to the owner of the **private key**. The public key can be thought of as an open padlock. Once closed only the private key can unlock it.
- The **secret key** is used only to decrypt messages (and to generate the public key). It should be kept secret. The secret key can be seen as a key that opens every padlock closed by the public key.


One of the most used asymmetric encryption scheme today is the <span style="color:red">RSA</span>.  It is based on some mathematical observations. Let's recall some arithmetical properties:
- A prime number is a whole number with exactly 2 distinct divisors. Every integer greater or equal than 2, that is not a prime number is called a composite number.
- Every whole number can be uniquely written as a product of prime numbers:
$$n=p^{\alpha_{1} }_{1}p^{\alpha_{2} }_{2}\  ...\  p^{\alpha_{k} }_{k}$$
- A product of two prime numbers is called a semi-prime number. It has 3 or 4 distinct divisors
(3 if it is a square of a prime number).


The basic idea of the RSA scheme follows: it easy to multiply 2 prime numbers to obtain a semi-prime number but it's difficult to factorize a semi-prime number into 2 prime numbers.
$$3\times 7=?\  \text{Easy !} $$
$$13\times 19=?\  \text{Less easy but alright } $$
$$ 187 = ? \times ?  \text{difficult } $$
$$ 186797 = ? \times ?  \text{Nasty af } $$


10016444466812877516651347592092877606999325867156134902126474870576401310371509197937849497 
32061828879847934816861684862864326449214280155473757303841537670351486905855745788953294686653 
05667852687855685298115910404311303180404287100354588108313006482467735715047743256036128648480 
80027194762485965856140145864228400743999303156570382089086775865731055296724143521221327468628 
21950171266360637073763193766827057457206146627252158883606266393926431447227342695628623860494 
83076188549980295606990827731968687429507788792780286440882172770001367957911700000685949637652 
38831914470509293382332669418868301436781248853885000370663778352581253239270257156871660127150 
01725765933851378635689651151763527144099274447723857372797474452663650725422387256011846500895 
56049862683135640206298862612679119720709968586034215160997260304220155673434151135668320749865 
84807932093124539029156912634836160456728007753201898072897827815590459999295298908223557195231 
58977763724441639028178539046224952247530731887239092769161189850803594847326119864462181341673 
60716012369946975020768242661592323585459972285070236101616423672439653172724479999925676798119 
71560093919447685551083829047142039685301977153590924844326332056772159786693521935447299870583


A human being cannot factorise this semi-prime number, nor can a machine at the time this paper is written. In 1991, the RSA laboratories published 50 semi-prime numbers (called the RSA numbers) from 100 to 2048 decimal digits. For some numbers, a reward was offered for the factorisation of the number. Some easy numbers were factorised quite quickly (the RSA-100 was factorised less than 2 weeks after the publication of the challenge) and some of them are still being factorised today (the example of the RSA-250 that was factorised in February 2020 is relevant) even if the challenge ended in 2007. The biggest RSA number to have ever been factorized is the RSA 768, that was factorized in 2009 using the General Number Field Sieve (GNFS).


The GNFS is the best-known algorithm for factorising integer (at least for large integers). The GNFS has an algorithmic complexity of 
$$\exp \left\{ \left[ \sqrt[3]{\frac{69}{4} } +O\left( 1\right)  \right]  \left( \text{ln} \left( n\right)  \right)^{\frac{1}{3} }  \left[ ln\left( ln\left( n\right)  \right)  \right]^{\frac{2}{3} }  \right\}  $$


where $n$ is the integer to factorise. As we won't discuss much more of how the GNFS works, the interested reader may refer to for a discussion on the algorithm and the complexity. The GNFS is one the best algorithm to factorise integer, but, nevertheless, it would take more than a billion years to factorise a 2048 digit key. In this sense, the RSA is a secure protocol.

I would like to emphasise here that the security of the RSA scheme is based on a computation- ally hard mathematical problem and the belief that we won't find another more efficient and much quicker algorithm to solve the problem. But we are believing in the security of the RSA because it has survived 50 years of attacks.


## Quantum cryptography


In the early '80s, the idea of a quantum computer began to grow and with it, the hope to compute and simulate things that were not possible or will ever be possible on a classical computer [6, 7]. In particular, FEYNMAN proposed to simulate Physics with a quantum computer, that he didn't assimilate to the quantum analog of a Turing machine but rather, to a universal quantum simulator. BENIOFF, in his paper, explored the possibility of a quantum Turing machine, or universal quantum computer. Without speaking more about the idea of a quantum Turing machine, we will see that we are today far from having such a computer.

One of the main advantages of quantum information is the superposition principle. The qubit, the analog of the classical bit, can take the value |0⟩ (analog of the bit 0), the value |1⟩ (analog of the bit 1), or a superposition of the two, meaning that with one qubit, we can store much more information than with one bit. We saw some vulgarisation programs that compared the classical bit to a switch. Then a qubit was compared to a switch that could be open, closed, and both at the same time. But this doesn't feel like a good comparison, and they often don't speak about the measurement.

The information encoded in the superposition of state, if not used correctly, will be destroyed when a measurement on the qubit is made.

There was also the hope of a quantum speedup, i.e. to compute or simulate things faster on a quantum computer than on a classical computer one even if the computation or simulation was possible in the first place.

One of the simplest examples is **GROVER's algorithm**. The basic idea of GROVER's algorithm is to find an element in a list. Supposing a list of length $N$, using a classical algorithm, the best case is to find the element in the first position, the worst case, in the $N$ th position, and on average, you need $\frac{N}{2}$ operations, which means a complexity in $O\left( N\right)$ . Using a technique called amplitude amplification, GROVER's algorithm has a complexity of $O\left( \sqrt{N} \right)  $

Another promising algorithm is **SHOR's algorithm**. It was introduced in 1997 by Peter SHOR
. The purpose of SHOR's algorithm is to factorize integers and has a complexity of

$$\left( \text{ln} \left( n\right)  \right)^{2}  \left( \text{ln} \left( \text{ln} \left( n\right)  \right)  \right)  \left( \text{ln} \left( ln\left( ln\left( n\right)  \right)  \right)  \right)  $$


This algorithm is much faster than the most efficient classical algorithm. I won't explain in detail how SHOR's algorithm works, but it's based on a classical algorithm with a quantum subroutine, that finds a period of a function, using Quantum Fourier Transform



Then, we might break the RSA in the foreseeable future. There are however several issues. Although Google claimed to have reached Quantum Supremacy, i.e. they claimed to have computed something that is not computable on classical computers (they estimated at 10 000 years the amount of time needed on the most powerful classical computers to compute what they did in 200 seconds), they have done a task that is quite resilient to errors. SHOR's algorithm is less error-resilient and the error rate that we have in quantum computers today is too high for factorising large integers. Today, the largest integer factorised by SHOR's algorithm is 21.

Nevertheless, we might see the RSA broken in the near future. What must be understood is that any encrypted message saved by a malicious party could be decrypted as soon as RSA is broken. This means that any private message sent today can become a public message later. To prevent those issues, some governments and companies are working in an area called **post- Quantum Cryptography**.


## Post-Quantum Cryptography

In 2016, the NATIONAL INSTITUTE OF STANDARDS AND TECHNOLOGY (NIST) of the US Department of Commerce issued a report on post-Quantum Cryptography. It first described what is the core of modern cryptography: public-key encryption, digital signatures, and key exchange. Some other components are required and we will talk of them later. Then the report continues by de- scribing how a potential quantum computer would be capable of breaking or at least weakening some of those algorithms. The RSA, as we saw, would become insecure and the AES (Advanced Encryption Standard) which is a widely used symmetric scheme to encrypt data would require larger keys (GROVER's algorithm can help breaking symmetric encryption).


The issue is the fact that those algorithms use some kind of mathematical problem that is hard to solve on a classical computer (factorising whole numbers or the discrete log problem) and that we could experience a quantum speedup to solve those problems.

Then there are two approaches to solve this problem:

- find mathematical problems that are impossible to solve in a reasonable amount of time even with a quantum computer
- use physical constraints, using for example quantum mechanics to design unconditionally secure protocols.

The first point is more on the classical side of Quantum Cryptography and that is usually what we mean when we talk about post-Quantum Cryptography. Some problems are already good candidates for such algorithms and some of them are already implemented. Indeed, even if SHOR's algorithm can break cryptosystems based on large integer factorisation or discrete logarithms, it can't be applied to some other cryptosystems like

- Hashed-based cryptography (for example **MERKLE's hash tree**)
- Code-based cryptography (**MCELIECE's hidden-Goppa-Code**)
- Lattice-based cryptography (**NTRU**)
- Multivariate-quadratic-equations cryptography
- Symmetric cryptography (recent symmetric cryptography schemes are considered pretty se- cure against quantum attacks)

**GROVER's algorithm** can provide help to break some of those schemes, but as the algorithm has a smaller speedup, it is sufficient to use a larger key. But we may one day find a quantum algorithm that can help breaking any of these cryptosystems (even if these schemes are believed to be quantum-safe and selected by the NIST).



# Quantum Voting

Overall articles

[Quantum voting scheme with multi-qubit entanglement](https://www.nature.com/articles/s41598-017-07976-1)
[Quantum protocols for anonymous voting and surveying](https://journals.aps.org/pra/abstract/10.1103/PhysRevA.75.012333)



## Electronic voting

Electronic voting refers to two types of voting methods:
- First, when electronic devices are placed at voting locations, i.e. under the responsibility of government officials
- Secondly, when internet voting is used, i.e. citizens can vote from wherever they want.

Two main requirements for voting were identified in: Why Electronic Voting Is Still A Bad Idea

{{< youtube LkH2r-sNjQs >}}

## Voting Requirements

A secure voting system must satisfy two fundamental requirements:

1. **Anonymity**: It must be impossible to identify how any individual voted after the election.
2. **Trustworthiness**: Each voter must be able to verify that their vote was properly counted.

These requirements are particularly challenging to meet with electronic voting systems, especially when voting is conducted over the internet.

## Quantum Voting

Following by the two articles above, we are going to specify in more detail the requirements of a voting scheme. An open ballot system is a voting scheme where users sign their ballot or make them identifiable in some way. In this system, each voter can know what an other voter has voted. For instance, a "show of hands" vote in a general assembly or when MPs cry "aye" or "no" in the UK Parliament are examples of open ballot systems. Opposed to those systems are the closed ballot systems, or anonymous systems, that have 3 main requirements:

- Correctness: a vote that is considered valid (i.e. with no
identifying mark and from a valid voter) should be
counted. A vote that is no considered valid should not be
counted
- Anonymity: Votes are anonymous (i.e. no identifying mark
on the ballot (*a system of voting secretly and in writing
on a particular issue*)
- Receipt-free: there should not be any way to know how a
voter voted or even if he has voted after the election.

A system of voting secretly and in writing on a particular issue. In the article [Quantum Voting Scheme Based on Conjugate Coding](https://www.ntt-review.jp/archive/ntttechnical.php?contents=ntr200801sp3.html)

the authors presented a voting scheme based on conjugate coding and uses the same bases and states that the BB84 protocol which is one of the first examples of plausible Quantum Cryptography the aim is to share a secret key without any pre-shared secret (or a very small one). This voting scheme requires:


- A counter, or scrutinizer $C$. He will verify and count votes. **He needs to be trusted.**

- An administrator$ A$. He will issue blank ballots. There is no need for total trust in the administrator as he can be supervised by $C$, or even split into several parties (each party can forge a part of the ballot)
-  $v$ voters $V_i (i = 1,...n)$
In the beginning, the administrator A forges blank ballots for every voter. We now explain what a blank ballot is:

We denote by $n$ a security parameter which is the length of a blank piece. A blank ballot is composed of several blank pieces.

At the beginning, the administrator randomly creates a secret $K$. This secret is the choice of $n+1$bases chosen between $B_x$ and $B_z$ :

                                     $$K = (B_{1}, B_{2}, \ldots, B_{n+1})$$

To forge a blank piece, the administrator randomly chooses $n$ bits  $b_{1},b_{2},b_{3},...b_{n}$ and $b_{n+1}$ is defined to be  $b_{n+1}=b_{1}\oplus b_{2}\oplus ...b_{n}$ . The bits ( $b_{1}...b_{n+1}$ ) re encoded into quantum states using the bases of the secret $K$

For instance, if $n=2$, with $K=(B_{x},B_{z},B_{x})$. A valid blank piece would be, using $b_{1}=1$ and $b_{2}=0$

                                      $$(|-\rangle, |0\rangle, |-\rangle)$$

To construct a blank ballot for voter, $V_i$ , the administrator constructs $m$ blank pieces with  $r^{i}_{1},...r^{i}_{m},$

                                     $$r^{i}_{j} = (b^{i}_{j,1}, \ldots, b^{i}_{j,n+1})$$

and  $$ $b^{i}_{j,\  n+1}=b^{i}_{j,1}\oplus ...\oplus b^{i}_{j,n}$ for $j=1,...,m.$


Next, one blank ballot is sent to each voter

When the voter receives his ballot, he will first randomize the ballot. This step is here to preserve the anonymity of the voter. This is done as follows: for each state of each blank piece of the blank ballot, the voter will apply either the identity or the gate $\sigma_{x} \sigma_{z}$ . For the n first states of the blank piece, the gate to apply is chosen at random. For the last bit of the blank piece, the gate is the identity, if the gate $\sigma_{x} \sigma_{z}$ was applied an even number of times on the $n$  first states and  $\sigma_{x} \sigma_{z}$ if the gate  $\sigma_{x} \sigma_{z}$ was applied an odd number of times, this will make the blank piece stay valid. The gate  $\sigma_{x} \sigma_{z}$ flips the state without changing its basis (the gate may change the overall phase but this cannot be measured).

$$\sigma_{x} \sigma_{z} |0\rangle = \sigma_{x} |0\rangle = |1\rangle$$
$$\sigma_{x} \sigma_{z} |1\rangle = -\sigma_{x} |0\rangle = -|0\rangle$$
$$\sigma_{x} \sigma_{z} |+\rangle = \sigma_{x} |-\rangle = -|-\rangle$$
$$\sigma_{x} \sigma_{z} |-\rangle = \sigma_{x} |+\rangle = |+\rangle$$
where $$|\pm \rangle = \frac{1}{\sqrt{2}} (|0\rangle \pm |1\rangle)$$  

By doing this step, the basis for each state remains the same, but the value if random. Also, the blank pieces remain valid.

Now that the voter has to cast his vote. We suppose that the set of all possible votes is a subset of $\left\{ 0,1\right\}^{m}$  . To write his vote, the voter applies either the identity (if he wants to encode a $0$) or the gate $\sigma_{x} \sigma_{z}$  (if he wants to encode a 1) to the last state of each blank piece.

Finally, the voter sends his vote to the counter  $C$. Once the counter has received all votes, the administrator sends $K$ to the counter through a secure classical channel. For each ballot, he measures each piece using $K$. The value of the bit for the piece is retrieved by summing all the results. The counter $C$ verifies that the results are indeed a possible vote. If it is, the vote is counted, and if not, the vote is discarded.

The security of this voting scheme relies on the fact that a malicious party cannot create a ballot with a valid choice with high probability without knowing the secret $K$. In fact, the probability for a ballot that was forged without $K$ to be accecpted is $\frac{Candidates}{2^{m}}$  . As the number of candidates is constant, the probability can be made arbitrarily small by adding pieces to the ballot.

Let's see a quick example before concluding this section. Let $n=3$  with two possible candidates: $010$ and $101$. The secret key is $K=\left( B_{z},B_{x},B_{x},B_{z}\right)$  

The administrator A forges the ballot:

$$(|1\rangle, |+\rangle, |+\rangle, |1\rangle)$$
$$(|0\rangle, |-\rangle, |+\rangle, |1\rangle)$$
$$(|0\rangle, |-\rangle, |-\rangle, |0\rangle)$$

$$(1,0,0,1), (0,1,0,1), (0,1,1,0)$$

The ballot is randomised by the voter:

$$(|1\rangle, |+\rangle, |+\rangle, |1\rangle)$$
$$(|0\rangle, |+\rangle, |+\rangle, -|0\rangle)$$
$$(|1\rangle, |-\rangle, |+\rangle, |0\rangle)$$

Applying the following the operators:

$$I \otimes I \otimes I \otimes I$$
$$I \otimes \sigma_{x} \sigma_{z} \otimes I \otimes \sigma_{z} \sigma_{x}$$
$$\sigma_{x} \sigma_{z} \otimes I \otimes \sigma_{x} \sigma_{z} \otimes I$$

The voter wants to vote for $101$ so he will flip the last states for the first and last pieces (the
blank ballot corresponds to $000$, and the final ballot is hence

$$(|1\rangle, |+\rangle, |+\rangle, -|0\rangle)$$
$$(|0\rangle, |+\rangle, |+\rangle, -|0\rangle)$$
$$(|1\rangle, |-\rangle, |+\rangle, |0\rangle)$$

Applying the following the operators:

$$I \otimes I \otimes I \otimes \sigma_{x} \sigma_{z}$$
$$I \otimes I \otimes I \otimes I$$
$$I \otimes I \otimes I \otimes \sigma_{x} \sigma_{z}$$

The counter C then makes measurements using the bases of K and finds the following results:

$$(1,0,0,0) \oplus_{2} = 1$$
$$(0,0,0,0) \oplus_{2} = 0$$
$$(0,1,1,1) \oplus_{2} = 1$$

where, $\oplus_{2} \$  is the sum modulo $2$ over all the results of the measurements of one piece. $C$ finds the vote  $101$ which is a correct vote and count it.




**Evaluation**

This protocol uses conjugate coding to implement a voting scheme. It is unconditionally guaranteed anonymous and has the property of correctness if the one-more unforgettability. This assumption is the following: there is no polynomial-time quantum algorithm that can create $l+1$ blank pieces out of $l$ blank pieces with high probability. However, to be implemented it requires quantum memory.

Also, the counter must be absolutely trusted. A version can remove this assumption: where a voter sends his vote to all other voters. The secret is shared with every voter after all the votes have been cast. This is however much heavier in quantum resources.

**[Preliminaries about the conjugate coding](https://dl.acm.org/doi/10.1145/1008908.1008920)**

*Conjugate coding* is one of the most important notions of Quantum Cryptography. The term is attributed to WIESNER, who proposed the idea in an unpublished (at first) and unnoticed article written in 1970, but was published in 1983 after Quantum Cryptography became a more plausible idea .


It is also called *quantum coding* and *quantum multiplexing*. The basic idea is that we can encode our classical bits 0 or 1 in different bases, and that a measurement in one of the basis will completely randomise the result in the other.

Consider for example the bases $B_z$ and $B_x$

$$B_{z} = \{|0\rangle, |1\rangle\}$$
$$B_{x} = \{|+\rangle, |-\rangle\}$$
$$|\pm \rangle = \frac{|0\rangle \pm |1\rangle}{\sqrt{2}}$$

Now imagine you want to encode the bit. If you do so in the $B_z$ basis, you will have the state $|1\rangle$  . Now if you measure this state in the $B_z$ basis, you will always obtain $|1\rangle$  and recover the good bit as

$$|\langle 0|1\rangle|^{2} = 0$$
$$|\langle 1|1\rangle|^{2} = 1$$

But, if you make a measurement in the $B_x$ basis, you will have the state $|+\rangle$  and recover the bit $0$, so the bad one) half of the time $|-\rangle$ (and recover the good bit) the other half since

$$|\langle +|1\rangle|^{2} = \frac{1}{2}$$
$$|\langle -|1\rangle|^{2} = \frac{1}{2}$$

We say that $B_z$ and $B_x$ are *conjugate* to each other.

This is the code for the BB84 protocol steps by steps:



```python
from random import randint
bits =[]
for i in range(8):
    bit = randint(0,1)
    bits.append(bit)

```
Output:  [0,  1,  1, 1,  0,  0,  0,  0]

```python
from random import choice
for i in range(8):
    base = choice(['X','Z'])
    basis.append(base)
basis
```


Output:  ['X',  'Z',  'X',  'X',  'X',  'Z', 'X',  'Z']
`

```python
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
q = QuantumRegister(8)
c = ClassicalRegister(8)
qc = QuantumCircuit(q,c)

for i in range(8):
    if basis[i]=='Z':
        if bits[i]==0:
            qc.i(q[i])
        else:
            qc.x(q[i])
    else:
        if bits[i]==0:
            qc.h(q[i])
        else:
            qc.x(q[i])
            qc.h(q[i])
qc.barrier()
qc.draw('mpl')

bobs_base =[]
for i in range(8):
    base = choice(['X','Z'])
    bobs_base.append(base)
bobs_base

```

```python

for i in range(8):
    if bobs_base[i]=='Z':
        qc.measure(q[i],c[i])
    else:
        qc.h(q[i])
        qc.measure(q[i],c[i])
qc.draw('mpl')

```


![image](/uploads/app11/outputa.png)


```python
#Running on Qsasimulator
from qiskit import transpile
from qiskit.providers.aer import QasmSimulator
backend = QasmSimulator()
qc_compiled = transpile(qc,backend)
job = backend.run(qc_compiled,shots = 1)
result = job.result()
counts = result.get_counts()
print(counts)
from qiskit.visualization import plot_histogram
plot_histogram(counts)
result = list(counts.keys())[0]
result
for i in range(8):
    if bobs_base[i]==basis[i]:
        print(result[i])
```

A nice web for the QKD with BB84 with steps can be read from here:

[Medium](https://tyagi-bhaumik.medium.com/quantum-key-distribution-harnessing-quantum-phenomena-for-secure-communication-9d9bb545f593)


## Pitch sum-up


**Introduction:**

In our quest for a sustainable future, every decision counts. From policy choices to technological advancements, the path we take today will shape the world we leave for future generations. One crucial aspect of this journey is ensuring that our voting systems are secure, transparent, and capable of fostering trust in the democratic process. In this context, we propose a revolutionary shift from traditional electronic voting systems to quantum voting schemes to support sustainable energy initiatives.

The Current Challenge:
Electronic voting systems have been the backbone of modern democracies for years. However, they come with inherent vulnerabilities, including the risk of cyberattacks and data breaches. In the subject of sustainable energy, where decisions can have profound global impacts, it is imperative that the voting process is beyond reproach and immune to manipulation. Quantum voting offers a paradigm shift in this direction.

**Quantum Voting: A Game-Changer for Sustainable Energy:**

1. Unprecedented Security:
Quantum voting leverages the laws of quantum mechanics to provide an unparalleled level of security. Unlike electronic voting systems, where vulnerabilities can be exploited, quantum voting relies on the fundamental principles of quantum entanglement and superposition, making it virtually impossible for malicious actors to tamper with the process.
2. Transparency and Verifiability:
Quantum voting ensures that each vote is transparently recorded and verified, enhancing trust in the electoral process. Quantum states, once measured, cannot be altered without detection, providing an immutable record of the vote.
3. Encouraging Global Collaboration:
Sustainable energy initiatives often require international cooperation. Quantum voting enables secure and tamper-proof cross-border voting, fostering trust among nations and facilitating collaborative efforts in the transition to sustainable energy sources.
4. Protecting Minority Voices:
In the pursuit of sustainable energy, it is crucial to consider the interests of all stakeholders, including minority groups. Quantum voting allows for secure and anonymous voting, ensuring that even marginalized voices are heard and respected.
5. Long-Term Viability:
Quantum technologies are on the rise and are poised to become an integral part of our future. By adopting quantum voting now, we future-proof our electoral systems and position ourselves at the forefront of technological innovation in the pursuit of sustainable energy.

**Conclusion:**
The transition to sustainable energy is a global imperative, and the decisions we make today will have far-reaching consequences. Quantum voting offers an innovative solution to the challenges of security, transparency, and trust in the democratic process, making it an ideal choice for shaping the future of sustainable energy initiatives. By embracing quantum voting, we not only safeguard the integrity of our elections but also demonstrate our commitment to a greener, more sustainable world. It's time to take the quantum leap for a sustainable energy future



### **References**

<a href="https://arxiv.org/pdf/1112.1212" style="color:#1E90FF;">
Rui-Rui Zhou, Li Yang. "Quantum election scheme based on anonymous
quantum key distribution" Chinese Physics B, Volume 21, Number 8.
</a>





### **About the author**



<div align="center">
  <img src="/uploads/app11/huybinh.png" alt="Author's Photo" width="150" style="border-radius: 50%; border: 2px solid #1E90FF;">
  <br>
  <strong>Huy Binh TRAN</strong>
  <br>
  <em>Master 2 Quantum Devices at Institute Paris Polytechnic, France</em>
  <br>
  <a href="https://www.linkedin.com/in/huybinhtran/" style="color:#1E90FF;">LinkedIn</a>
</div>





{{< math >}}
{{< /math >}} 