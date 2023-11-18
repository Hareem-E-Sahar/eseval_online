df<-read.csv("/home/hareem/UofA2023/eseval_v2/eseval_timewise_batched/all_irjit_results_on_sampled_commits_by_10.csv",header=TRUE)
dataK3 = df[df$cumulative=="Yes" & df$K==3,]
summary(dataK3$gmean)
sd(dataK3$gmean)

#IRJIT batched and JITLine 
df_jitline<-read.csv("/home/hareem/UofA2023/JITLine-replication-package/JITLine/all_jitline_results_on_sampled_commits_group_by10.csv",header=TRUE)
df_jitline <- df_jitline[df_jitline$cumulative=="Yes",]
wilcox.test(dataK3$gmean, df_jitline$gmean)
summary(df_jitline$gmean)
sd(df_jitline$gmean)

#IRJIT online and JITLine
irjit_online<-read.csv("/home/hareem/UofA2023/eseval_v2/eseval_timewise/all_irjit_results_on_all_commits_in_test_data.csv",header=TRUE)
irjit_online = irjit_online[irjit_online$cumulative=="Yes" & irjit_online$K==3,]
wilcox.test(irjit_online$gmean, df_jitline$gmean)


neutron = df2[df2$project=="neutron",]
summary(neutron$gmean)
tomcat = df2[df2$project=="tomcat",]
summary(tomcat$gmean)
nova =  df2[df2$project=="nova",]
summary(nova$gmean)
broadleaf =  df2[df2$project=="BroadleafCommerce",]
summary(broadleaf$gmean)
fabric =  df2[df2$project=="fabric8",]
summary(fabric$gmean)
jgroups =  df2[df2$project=="JGroups",]
summary(jgroups$gmean)
spring =  df2[df2$project=="spring-integration",]
summary(spring$gmean)
camel =  df2[df2$project=="camel",]
summary(camel$gmean)
npm = df2[df2$project=="npm",]
summary(npm$gmean)
brackets = df2[df2$project=="brackets",]
summary(brackets$gmean)



neutron2 = df_jitline[df_jitline$project=="neutron",]
summary(neutron2$gmean)
tomcat2 = df_jitline[df_jitline$project=="tomcat",]
summary(tomcat2$gmean)
nova2 =  df_jitline[df_jitline$project=="nova",]
summary(nova2$gmean)
broadleaf2 =  df_jitline[df_jitline$project=="BroadleafCommerce",]
summary(broadleaf2$gmean)
fabric2 =  df_jitline[df_jitline$project=="fabric8",]
summary(fabric2$gmean)
jgroups =  df_jitline[df_jitline$project=="JGroups",]
summary(jgroups$gmean)
spring2 =  df_jitline[df_jitline$project=="spring-integration",]
summary(spring2$gmean)
camel2 =  df_jitline[df_jitline$project=="camel",]
summary(camel2$gmean)
npm2 = df_jitline[df_jitline$project=="npm",]
summary(npm2$gmean)
brackets2 = df_jitline[df_jitline$project=="brackets",]
summary(brackets2$gmean)


#If the p-value is less than your significance level, you would reject the null 
#hypothesis, suggesting there's a statistically significant difference in the 
#distributions of the two groups. 


#Wilcoxon rank sum test with continuity correction
#data:  dataK3$gmean and df_jitline$gmean
#W = 733605, p-value = 0.4439
#alternative hypothesis: true location shift is not equal to 0

#we reject null if p<0.05
#p=0.4439 so can't reject null for batched irjit and jitline.
#there is no difference.
#Otherwise, you would fail to reject the null 
#hypothesis, suggesting there's no significant difference.


#Wilcoxon rank sum test with continuity correction
#data:  irjit_online$gmean and df_jitline$gmean
#W = 8792352, p-value = 0.5253
#alternative hypothesis: true location shift is not equal to 0


#we reject null if p<0.05
#p=0.5253 so can't reject null for online irjit and batched jitline.
#there is no difference.
#Otherwise, you would fail to reject the null 
#hypothesis, suggesting there's no significant difference.
