# Bruger skelletet af AAN-v1.py som er en oversættelse og re-skrivning af https://github.com/mnielsen/neural-networks-and-deep-learning/blob/master/src/network.py 

#-#-#-#-# Værktøjer

from Laantefiler import mnist_loader 
from Laantefiler import expand_mnist
#-#-#-#-# Biblioteker

# Standard biblioteker

# Tredjeparts Biblioteker
import numpy as np

#-#-#-#-# Ekstra matematiske funktioner fra https://github.com/mnielsen/neural-networks-and-deep-learning/blob/master/src/network.py

def sigmoid(z):
        """Sigmoid funktionen."""
        return 1.0/(1.0+np.exp(-z))

def afledt_sigmoid(z):
    """Den afledte Sigmoid funktionen."""
    return sigmoid(z)*(1-sigmoid(z))

#-#-#-#-# Tilføjede ekstra matematiske funktioner

def ReLU(z):
    """ReLU activation function."""
    return np.maximum(0, z)

def afledt_ReLU(z):
    """Afledt ReLU."""
    return np.where(z > 0, 1, 0)

def LeakyReLU(z, alpha=0.01):
    """Leaky ReLU aktiverings function."""
    return np.where(z > 0, z, alpha * z)

def afledt_LeakyReLU(z, alpha=0.01):
    """Afledt Leaky ReLU."""
    return np.where(z > 0, 1, alpha)

#-#-#-#-# Netværk basseret på https://github.com/mnielsen/neural-networks-and-deep-learning/blob/master/src/network.py (Evt. tifløjelser vil være tydligt markeret)

class netværk:
    def __init__(self, sizes: list):
        '''
        ``sizes`` er in liste af lag:
        F.eks: [3,2,3] er et netværk med 3 input neuroner, et skjult lag med 2 neuroner og et outputlag med 3 neuroner.

        '''
        # Gemmer information om netværket
        self.lag = len(sizes)
        self.sizes = sizes
        # Laver tilfældige biaser og vægte (laver et array for hvert lag af tilfældige tal (Gaussian distribution) for hvert neuron i hvert lag)
        #For at justere vægtende såde ikke bliver for høje bliver bægtene yderligere formindsket (ReLU tilføjelse)
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]] 
        self.weights = [np.random.randn(y, x) * np.sqrt(2 / x) for x, y in zip(sizes[:-1], sizes[1:])]
    
    def feedforward(self, a:list):
        """Retunerer outputet af netværket hvis ``a`` er input"""
        for b,w in zip(self.biases, self.weights):
            #print(f"Vægt: {w}, Bias: {b}, Input {a}")
            a = LeakyReLU(np.dot(w, a) + b)

        return a
    
    def SGD(self, trænings_data:list, epochs:int, mini_batch_size:int, eta:float, test_data=None):
        """
        Træner netværket med stochastic gradient descent, ved at gøre brug af mini-batches.
        ``trænings_data`` er en liste af tuples som viser input og forvetet output: ``(x,fx)``.
        Hvis ``test_data`` er andgivet ville netværket blive evalueret udfra den data.
        
        """ 
        trænings_data = list(trænings_data) # Bare fot at være sikker
        n = len(trænings_data)

        if test_data:
            test_data = list(test_data)# Bare fot at være sikker
            n_test = len(test_data)

        for j in range(epochs):
            mini_batches = [trænings_data[k:k+mini_batch_size]
                            for k in range(0, n, mini_batch_size)] # Opdeler data i lister af mini_batch_size'te størrelser
            for batch in mini_batches:
                self.updater_mini_batch(batch, eta)
                

            if test_data:
                print("Epoch {} : {} / {}".format(j,self.evaluer(test_data),n_test))
                
            else:
                print("Epoch {} complete".format(j))
            
        
    
    def updater_mini_batch(self, mini_batch, eta):
        """
        Updaterer netværkets vægte, ``self.weights``, og biaser,``self.biases``, ved at bruge gradient descent og backpropagation af et mini-batch.
        ``mini_batch``-et er en liste af af tuples som viser input og forvetet output: ``(x,fx)`` og ``eta`` er læringsraten.
        """
        gradient_b= [np.zeros(b.shape) for b in self.biases]
        gradient_w = [np.zeros(w.shape) for w in self.weights]

        for x, fx in mini_batch:
            fejlsignal_gradient_b, fejlsignal_gradient_w = self.backprop(x,fx)
            gradient_b = [gb+fgb for gb,fgb in zip(gradient_b, fejlsignal_gradient_b)]
            gradient_w = [gw+fgw for gw,fgw in zip(gradient_w, fejlsignal_gradient_w)]
        self.weights = [w-(eta/len(mini_batch))*gw
                        for w, gw in zip(self.weights, gradient_w)]
        self.biases = [b-(eta/len(mini_batch))*gb
                       for b, gb in zip(self.biases, gradient_b)]
        
        
             
    def backprop(self, x, fx):
        """
        Retunerer en  tuple ``(gradient_b, gradient_w)`` som er gradiente for vores fejlfunktion C_x.
        ``gradient_b`` and ``gradient_w`` er layer-by-layer lister af numpy arrays, ligesom
         ``self.biases`` og ``self.weights``.
        
        """
        gradient_b= [np.zeros(b.shape) for b in self.biases]
        gradient_w = [np.zeros(w.shape) for w in self.weights]

        a_vec = x
        a_list = [x] 
        z_list = []

        #-#-#-#-# Aktiverings funktionen sigmoid (og afledt_sigmoid) er blevet ændrert til LeakyReLU (og afledt_LeakyReLU)

        for b, w in zip(self.biases, self.weights):
            z = np.dot(w,a_vec)+b
            z_list.append(z)
            a_vec = LeakyReLU(z)
            a_list.append(a_vec)
        fejlsignal = self.afledt_fejl(a_list[-1], fx) * afledt_LeakyReLU(z_list[-1])
        gradient_b[-1] = fejlsignal
        gradient_w[-1] = np.dot(fejlsignal, a_list[-2].transpose())

        for l in range(2,self.lag):
            z_vec = z_list[-l]
            fejlsignal  = np.dot(self.weights[-l+1].transpose(), fejlsignal) * afledt_LeakyReLU(z_vec)
            gradient_b[-l] = fejlsignal
            gradient_w[-l] = np.dot(fejlsignal, a_list[-l-1].transpose())
        return (gradient_b, gradient_w) # retunerer radient_b og gradient_w
    
    def evaluer(self, test_data):
        """
        Evaluerer koden på bagrund af ``test_data``. Retunerer mægnden af tests hvor resultatet stemmer overens med det rigtige tal. 
        (Vi antager at resultatet er det tal hvor neuronens output/aktivering er størst)
        """
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)
    
    def afledt_fejl(self, a_list, fx):
        """Returner vectoren af de partielle afledede \partial C_x /
        \partial a for output listen a_list."""
        return (a_list-fx)



if __name__== "__main__":
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    net = netværk([784, 1000,10])
    net.SGD(training_data, 30, 10, 0.1, test_data=test_data) 