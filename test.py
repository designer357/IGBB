Iterations= 10
a=[90.837	,91.288	,91.469	,91.513	,91.062	,91.062	,91.288	,91.062	,91.288	,90.837]
for tab in range(Iterations):
    deviation_auc=0.0
    mean_auc=Compute_average_list(Temp_SubFeature_Auc_list[eachMethod+"_BN_"+str(bagging_number)+"_TF_"+str(Top_K)])
    for tab in range(len(Temp_SubFeature_Auc_list[eachMethod+"_BN_"+str(bagging_number)+"_TF_"+str(Top_K)])):
        temp = Temp_SubFeature_Auc_list[eachMethod+"_BN_"+str(bagging_number)+"_TF_"+str(Top_K)][tab]
        deviation_auc=deviation_auc+((temp-mean_auc)*(temp-mean_auc))
    deviation_auc/=Iterations