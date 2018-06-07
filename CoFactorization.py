
def MatrixSparseCoFactorization(X1,X2, latent_features, lamda0 = 0 ,lamda1 = 0,lamda2 = 0,max_iter=1000, error_limit=0.001, fit_error_limit=0.01):
    """
    Decompose X1 to W*H1 and X2 to W*H2
    """
    eps = 0.0001
    print 'Starting NMF co-decomposition with {} latent features and {} iterations.'.format(latent_features, max_iter)
     
    rows = len(X1)
    
    W = np.random.rand(rows, latent_features)
    W = np.maximum(W, eps)  
    #print X1.shape
    #print X2.shape  
    
    H1 = linalg.lstsq(W, X1)[0]
    H1 = np.maximum(H1, eps)
    
    H2 = linalg.lstsq(W, X2)[0]
    H2 = np.maximum(H2, eps)
    
    X1_est_prev = dot(W, H1)    
    X2_est_prev = dot(W, H2)
    lastFit = 0;
    
    for i in range(1, max_iter + 1):
        #update W
        top = dot(X1,H1.T) + dot(X2,H2.T)
        #bot = dot(dot(W,H1),H1.T) + dot(dot(W,H2),H2.T)
        bot = dot(W,dot(H1,H1.T) + dot(H2,H2.T) +  lamda0*np.identity(latent_features))
        W *= top/bot
        W = np.maximum(W, eps)
#      
        #update H1
        top  = dot(W.T,X1)
        bot = dot(dot(W.T,W)+ lamda1*np.identity(latent_features),H1 )
        H1*= top/bot
        H1 = np.maximum(H1, eps)
        
        #update H2
        top  = dot(W.T,X2)
        #bot = dot(W.T,dot(W,H2) + lamda2*np.identity(latent_features))
        bot = bot = dot(dot(W.T,W)+ lamda1*np.identity(latent_features),H2)
        H2*= top/bot
        H2 = np.maximum(H2, eps)       

        #==== evaluation ==== and stop condition
        
        if i % 5 == 0 or i == 1 or i == max_iter:
            print 'Iteration {}:'.format(i),
            X1_est = dot(W, H1)
            X2_est = dot(W,H2)
            err1 = (X1_est_prev - X1_est)
            err2 = (X2_est_prev - X2_est)
            err3 = W
            err4 = H1
            err5 = H2
            fit_residual = np.sqrt(np.sum(err1 ** 2)) + np.sqrt(np.sum(err2 ** 2))+ lamda0*np.sqrt(np.sum(err3 ** 2)) + lamda1*np.sqrt(np.sum(err4 ** 2)) + lamda2*np.sqrt(np.sum(err5 ** 2)) 
            X1_est_prev = X1_est
            X2_est_prev = X2_est
 
            #curRes = linalg.norm(mask * (X - X_est), ord='fro')
            print 'fit residual', np.round(fit_residual, 4)
            #print '\n'
            #print 'total residual', np.round(curRes, 4)
            #if math.fabs(fit_residual - lastFit)<eps:
                #break;
            if(math.fabs(fit_residual-lastFit)<eps):
                print lastFit;
                break;        
            lastFit = fit_residual
            if fit_residual < fit_error_limit:
                break
    #return W,H1,H2
    return normalization_Matrix_L1(W, H1, H2);
    #return normalization_Matrix_L2(W, H1, H2);
	
eps = 0.01
matrix = np.random.rand(300, 30) #250 is term and 30 is sentence
matrix = np.maximum(matrix, eps)    
#  
matrix1 = np.random.rand(300,100)
matrix1 = np.maximum(matrix1, eps)   
W,H1,H2 = MatrixSparseCoFactorization(matrix,matrix1,6);
print W.shape
print H1.shape
print H2.shape