#include <ecos.h>
#include <Eigen/SparseCore>

#define PI 3.14159265


// Norm Ball Barrier Function 
// float alpha = 0.15; 
// float C = 0.5; 
// float beta = 1; 

// Norm ball with heading
float alpha = 0.3;
float C = 0.33; 
float beta = 0.1; 
int turn = 0; 

float ballCBF(Eigen::Vector3f rho){
    return 1.0/2*(pow(rho.norm(),2)-pow(C,2)); 
}

float turnCBF(Eigen::Vector3f rho){
    return 1.0/2*(pow(rho.norm(),2) - pow(C,2)  - beta*( pow((PI/4), 2) - pow(atan(rho[0]/rho[2]),2))   );
}


// void argmin_cbf(std::vector<float> k_des, Eigen::Vector3f rho , std::vector<float> &u_star){
//     // double lower_bound = -alpha*CBF(rho)/(-rho[2]); 
//     // if (k_des[0] < lower_bound){
//     //     return k_des[0]; 
//     // }else{
//     //     return lower_bound; 
//     // }

//     Eigen::Vector2f Lgh = {-rho[2] + beta*atan(rho[0]/rho[2])*rho[0]/(pow(rho[2],2) + pow(rho[0],2)), -beta*atan(rho[0]/rho[2])};
    
//     double ineq = Lgh[0]*k_des[0] + Lgh[1]*k_des[1] + alpha*CBF(rho);
//     if (ineq > 0 ){
//         u_star[0] = k_des[0]; 
//         u_star[1] = k_des[1]; 
//     }else{
//         u_star[0]  = -Lgh[0]*alpha*CBF(rho)/(Lgh.dot(Lgh));
//         u_star[1]  = -Lgh[1]*alpha*CBF(rho)/(Lgh.dot(Lgh));
//     }
// }


// ECOS Type Defs 
typedef Eigen::SparseMatrix<double, Eigen::ColMajor> SpMat; // declares a column-major sparse matrix type of double
typedef Eigen::Triplet<double> T;


// ECOS PARAMS 
idxint n = 4; // number of decision vars
idxint p = 0; // number of equality constraints
idxint l = 0; // number of positive orthant constriants
idxint e = 0; // number of exponential cones 
pfloat c[4] = {1.0, 0.0, 0, 0};

// Uncertainty Params
int Lfh = 0; 
double L_lgh = 0; 
double err_epsilon = 0; 
double vel_weight = 5;

// void argmin_cbf_ecos(std::vector<float> k_des, Eigen::Vector3f rho , std::vector<float> &u_star){
//     // ECOS Params 
//     idxint m = 7; // number of constraints
//     idxint ncones = 2; // number of second order cones 
//     idxint q[2] = {4, 3}; // Array of the length of each of the ncones 

//     Eigen::Vector2f Lgh = {-rho[2] + beta*atan(rho[0]/rho[2])*rho[0]/(pow(rho[2],2) + pow(rho[0],2)), -beta*atan(rho[0]/rho[2])};

//     c[2] = -pow(vel_weight,2)*k_des[0]; 
//     c[3] = -k_des[1]; 
//     pfloat h_ecos[7] = {1/sqrt(2), -1/sqrt(2), 0.0, 0.0, alpha*CBF(rho) + Lfh , 0.0, 0.0};

// 	SpMat G(7,4); 
//     std::vector<T> tripletList;
//     tripletList.push_back(T(0,0,-1.0/sqrt(2))); 
// 	tripletList.push_back(T(1,0,-1.0/sqrt(2))); 
// 	tripletList.push_back(T(2,2,-1*vel_weight)); 
// 	tripletList.push_back(T(3,3,-1)); 
// 	tripletList.push_back(T(4,2,-Lgh[0])); 
// 	tripletList.push_back(T(4,3,-Lgh[1])); 
// 	tripletList.push_back(T(5,2,-L_lgh*err_epsilon)); 
// 	tripletList.push_back(T(6,3,-L_lgh*err_epsilon)); 
//     G.setFromTriplets(tripletList.begin(), tripletList.end()); 
// 	G.makeCompressed();
// 	double *Gpr = G.valuePtr(); 
// 	idxint Gir[8];
// 	idxint Gjc[5]; 

//     // Convert Type
// 	for(int i=0; i<8; i++){
// 		Gir[i] = (long)G.innerIndexPtr()[i]; 
// 	}
// 	for(int i=0; i<5; i++){
// 		Gjc[i] = (long)G.outerIndexPtr()[i]; 
// 	}

//     pwork *ecos_problem; 
// 	ecos_problem = ECOS_setup(n, m, p, l, ncones, q, e, Gpr, Gjc, Gir, NULL, NULL, NULL, c, h_ecos, NULL);

//     if (ecos_problem == NULL){
//         std:: cout << "Ecos setup problem " << std::endl; 
//     }

// 	idxint ecos_flag = ECOS_solve(ecos_problem);

//     if (ecos_flag == 0 || ecos_flag == 10){
//         u_star[0] = ecos_problem->x[2]; 
//         u_star[1] = ecos_problem->x[3]; 
//     }else{
//         std::cout << "ECOS Problem " << std::endl; 
//         std::cout << "ECOS flag = " << ecos_flag << std::endl;
//     }

//     ECOS_cleanup(ecos_problem, 0); 

// }



void argmin_cbf_ecos_multiple_points(const std::vector<float> & k_des,
                                     const std::vector<Eigen::Vector3f> & rhos ,
                                     std::vector<float> &u_star){

    idxint m = 4 + 3*rhos.size(); // Change number of constraints, 4 for the rotated cone and 3 for each CBF 
    idxint ncones = 1 + rhos.size(); // number of second order cones, the rotated cone + one for each point
    
    idxint* q = NULL; 
    pfloat* h_ecos = NULL; 

    q = new idxint[ncones]; 
    h_ecos = new pfloat[m]; 

    for (int i=0; i<ncones; i++){ 
        if (i==0){ 
            q[i] = 4; 
        }else{
            q[i] = 3; 
        }
    }




    c[2] = -pow(vel_weight,2)*k_des[0]; 
    c[3] = -k_des[1]; 
    h_ecos[0] = 1/sqrt(2);
    h_ecos[1] = -1/sqrt(2); 
    h_ecos[2] = 0.0;
    h_ecos[3] = 0.0; 


    SpMat G(m,4); 
    std::vector<T> tripletList;
    tripletList.push_back(T(0,0,-1.0/sqrt(2))); 
	tripletList.push_back(T(1,0,-1.0/sqrt(2))); 
	tripletList.push_back(T(2,2,-1*vel_weight)); 
	tripletList.push_back(T(3,3,-1)); 

    for(size_t rhs_idx=0; rhs_idx < rhos.size(); rhs_idx++){

        double xhat = rhos[rhs_idx][2]; 
        double yhat = rhos[rhs_idx][0];
        // Eigen::Vector2f Lgh = {-xhat + beta*atan(yhat/xhat)*yhat/(pow(xhat,2) + pow(yhat,2)), -beta*atan(yhat/xhat)}; 
        Eigen::Vector2f Lgh = {-1, 0};

        // h_ecos[4 + 3*rhs_idx] = alpha*CBF(rhos[rhs_idx]); 
        h_ecos[4 + 3*rhs_idx] = alpha*ballCBF(rhos[rhs_idx])/xhat; 
        h_ecos[4 + 3*rhs_idx + 1] = 0; 
        h_ecos[4 + 3*rhs_idx + 2] = 0; 

        tripletList.push_back(T(4 + 3*rhs_idx,2,-Lgh[0])); 
        tripletList.push_back(T(4 + 3*rhs_idx,3,-Lgh[1])); 
        tripletList.push_back(T(4 + 3*rhs_idx + 1,2,-L_lgh*err_epsilon)); 
        tripletList.push_back(T(4 + 3*rhs_idx + 2,3,-L_lgh*err_epsilon)); 
    }




    G.setFromTriplets(tripletList.begin(), tripletList.end()); 
	G.makeCompressed();
	double *Gpr = G.valuePtr(); 
	idxint Gir[m + rhos.size()]; // 1 for each constraint row plus an extra for each row with Lgh (of which there's one per every rho)
	idxint Gjc[5]; 

    // Convert Type
	for(size_t i=0; i<m + rhos.size(); i++){
		Gir[i] = (long)G.innerIndexPtr()[i]; 
	}
	for(int i=0; i<5; i++){
		Gjc[i] = (long)G.outerIndexPtr()[i]; 
	}

    // std::cout << G << std::endl; 

    pwork *ecos_problem; 
	ecos_problem = ECOS_setup(n, m, p, l, ncones, q, e, Gpr, Gjc, Gir, NULL, NULL, NULL, c, h_ecos, NULL);

    if (ecos_problem == NULL){
        std:: cout << "Ecos setup problem " << std::endl; 
    }

	idxint ecos_flag = ECOS_solve(ecos_problem);

    if (ecos_flag == 0 || ecos_flag == 10){
        u_star[0] = ecos_problem->x[2]; 
        u_star[1] = ecos_problem->x[3]; 
    }else{
        std::cout << "ECOS Problem " << std::endl; 
        std::cout << "ECOS flag = " << ecos_flag << std::endl;
    }

    ECOS_cleanup(ecos_problem, 0); 

    delete [] q; 
    q = NULL; 
    delete [] h_ecos; 
    h_ecos = NULL; 

}





void argmin_turn_cbf_ecos_multiple_points(const std::vector<float> & k_des,
                                     const std::vector<Eigen::Vector3f> & rhos ,
                                     std::vector<float> &u_star){

    idxint m = 4 + 3*rhos.size(); // Change number of constraints, 4 for the rotated cone and 3 for each CBF 
    idxint ncones = 1 + rhos.size(); // number of second order cones, the rotated cone + one for each point
    
    idxint* q = NULL; 
    pfloat* h_ecos = NULL; 

    q = new idxint[ncones]; 
    h_ecos = new pfloat[m]; 

    for (int i=0; i<ncones; i++){ 
        if (i==0){ 
            q[i] = 4; 
        }else{
            q[i] = 3; 
        }
    }




    c[2] = -pow(vel_weight,2)*k_des[0]; 
    c[3] = -k_des[1]; 
    h_ecos[0] = 1/sqrt(2);
    h_ecos[1] = -1/sqrt(2); 
    h_ecos[2] = 0.0;
    h_ecos[3] = 0.0; 


    SpMat G(m,4); 
    std::vector<T> tripletList;
    tripletList.push_back(T(0,0,-1.0/sqrt(2))); 
	tripletList.push_back(T(1,0,-1.0/sqrt(2))); 
	tripletList.push_back(T(2,2,-1*vel_weight)); 
	tripletList.push_back(T(3,3,-1)); 

    for(size_t rhs_idx=0; rhs_idx < rhos.size(); rhs_idx++){

        double xhat = rhos[rhs_idx][2]; 
        double yhat = rhos[rhs_idx][0];
        Eigen::Vector2f Lgh = {-xhat + beta*atan(yhat/xhat)*yhat/(pow(xhat,2) + pow(yhat,2)), -beta*atan(yhat/xhat)}; 
        // Eigen::Vector2f Lgh = {-1, 0};

        h_ecos[4 + 3*rhs_idx] = alpha*turnCBF(rhos[rhs_idx]); 
        // h_ecos[4 + 3*rhs_idx] = alpha*ballCBF(rhos[rhs_idx])/xhat; 
        h_ecos[4 + 3*rhs_idx + 1] = 0; 
        h_ecos[4 + 3*rhs_idx + 2] = 0; 

        tripletList.push_back(T(4 + 3*rhs_idx,2,-Lgh[0])); 
        tripletList.push_back(T(4 + 3*rhs_idx,3,-Lgh[1])); 
        tripletList.push_back(T(4 + 3*rhs_idx + 1,2,-L_lgh*err_epsilon)); 
        tripletList.push_back(T(4 + 3*rhs_idx + 2,3,-L_lgh*err_epsilon)); 
    }




    G.setFromTriplets(tripletList.begin(), tripletList.end()); 
	G.makeCompressed();
	double *Gpr = G.valuePtr(); 
	idxint Gir[m + rhos.size()]; // 1 for each constraint row plus an extra for each row with Lgh (of which there's one per every rho)
	idxint Gjc[5]; 

    // Convert Type
	for(size_t i=0; i<m + rhos.size(); i++){
		Gir[i] = (long)G.innerIndexPtr()[i]; 
	}
	for(int i=0; i<5; i++){
		Gjc[i] = (long)G.outerIndexPtr()[i]; 
	}

    // std::cout << G << std::endl; 

    pwork *ecos_problem; 
	ecos_problem = ECOS_setup(n, m, p, l, ncones, q, e, Gpr, Gjc, Gir, NULL, NULL, NULL, c, h_ecos, NULL);

    if (ecos_problem == NULL){
        std:: cout << "Ecos setup problem " << std::endl; 
    }

	idxint ecos_flag = ECOS_solve(ecos_problem);

    if (ecos_flag == 0 || ecos_flag == 10){
        u_star[0] = ecos_problem->x[2]; 
        u_star[1] = ecos_problem->x[3]; 
    }else{
        std::cout << "ECOS Problem " << std::endl; 
        std::cout << "ECOS flag = " << ecos_flag << std::endl;
    }

    ECOS_cleanup(ecos_problem, 0); 

    delete [] q; 
    q = NULL; 
    delete [] h_ecos; 
    h_ecos = NULL; 

}





// double get_L_lgh(Eigen::Vector3f rho, int dmin){ 
//     double du = rho[0]/rho[2]*focus; 
//     double dv = rho[1]/rho[2]*focus; 
//     double phi = -beta*dv/2/(pow(dv, 2) + pow(du,2)); 

//     Eigen::Vector2f dLgh = {du*baseline/pow(dmin, 2) + phi, phi }; 
//     // Eigen::Vector2f dLgh = 0; //{du*baseline/pow(dmin, 2) + phi, phi }; 

//     return dLgh.norm(); 
// }


// double get_L_ah(Eigen::Vector3f rho, int dmin){
//     double du = rho[0]/rho[2]*focus; 
//     double dv = rho[1]/rho[2]*focus; 
//     return abs(-pow(baseline,2)/pow(dmin,3)*(pow(du,2) + pow(dv,2)));
//     // return alpha * ball / xhat 
// }
