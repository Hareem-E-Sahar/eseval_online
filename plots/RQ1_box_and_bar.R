library(tidyr)
library(dplyr)
library(ggplot2)
data_online1<-read.csv("/home/hareem/UofA2023/eseval_v2/eseval_timewise_v1/results/metrics.csv",header=TRUE)
data_online2<-read.csv("/home/hareem/UofA2023/eseval_v2/eseval_timewise/results/metrics.csv",header=TRUE)
data_online<-rbind(data_online1,data_online2)
data_online = data_frame(project=data_online$project, time=data_online$time, approach="irjit_online")

data_batched1<-read.csv("/home/hareem/UofA2023/eseval_v2/eseval_timewise_batched/results_commit_level/results_lines_added_camel/metrics_v2.csv")
data_batched1 = data_frame(project=data_batched1$project, time=data_batched1$time, approach="irjit_batched")

data_batched2<-read.csv("/home/hareem/UofA2023/eseval_v2/eseval_timewise_batched/results_commit_level/results_lines_added_camel/metrics_v1.csv")
data_batched2 = data_frame(project=data_batched2$project, time=data_batched2$time, approach="irjit_batched")

data_batched3<-read.csv("/home/hareem/UofA2023/eseval_v2/eseval_timewise_batched/results_commit_level/results_lines_added_deleted_camel/metrics.csv")
data_batched3 = data_frame(project=data_batched3$project, time=data_batched3$time, approach="irjit_batched")

data_batched4<-read.csv("/home/hareem/UofA2023/eseval_v2/eseval_timewise_batched/results_commit_level/results_lines_added_deleted_shingle/metrics.csv")
data_batched4 = data_frame(project=data_batched4$project, time=data_batched4$time, approach="irjit_batched")

data_batched<-rbind(data_batched1,data_batched2,data_batched3,data_batched4)

df_jitline<-read.csv("/home/hareem/UofA2023/eseval_v2/plots/RQ1/jitline_total_train_time.csv",header=TRUE)
long_jitline <- gather(df_jitline, key = "experiment", value = "time", prep_rq4, run1, run2, run3, run4)
long_jitline = data_frame(project=long_jitline$project,time=long_jitline$time,approach="jitline_batched")

result_df<-rbind(long_jitline, data_online, data_batched)

my_boxplot <- ggplot(result_df, aes(x = project, y = time, fill = approach)) +
  geom_boxplot(alpha = 0.75)+
  theme_minimal() +
  labs(y="Run time in sec", x=element_blank(), fill=element_blank()) +  
  scale_fill_manual(values=c("irjit_online"="gray", "irjit_batched"="lightblue", "jitline_batched"="lightgreen"),
                    labels=c("irjit_online"=expression("IRJIT"["online"]), "irjit_batched"=expression("IRJIT"["batched"]),"jitline_batched"= expression("JITLine"["batched"]))) +  
  theme(axis.text.x=element_text(angle=45, hjust=1),
        legend.position="right",
        panel.grid.major = element_blank()) 

#write.csv(result_df, "/home/hareem/UofA2023/eseval_v2/plots/RQ1/all.csv")
average_data <-result_df %>%
  group_by(project, approach) %>%  
  summarise(average = mean(time, na.rm = TRUE), .groups = "drop")  

my_barplot <- ggplot(average_data, aes(x = project, y = average, fill = approach)) +
  geom_bar(stat = "identity", position = position_dodge(), alpha = 0.75)  +
  theme_minimal() +
  labs(y="Run time in sec", x=element_blank(), fill=element_blank()) +  
  scale_fill_manual(values=c("irjit_online"="gray", "irjit_batched"="lightblue", "jitline_batched"="lightgreen"),
                    labels=c("irjit_online"=expression("IRJIT"["online"]), "irjit_batched"=expression("IRJIT"["batched"]),"jitline_batched"= expression("JITLine"["batched"]))) +  
  theme(axis.text.x=element_text(angle=45, hjust=1),
        legend.position="right",
        panel.grid.major = element_blank()) 

pdf(file="/home/hareem/UofA2023/eseval_v2/plots/RQ1/RQ1_time_barplot.pdf")
my_barplot
dev.off()

pdf(file="/home/hareem/UofA2023/eseval_v2/plots/RQ1/RQ1_time_boxplot.pdf")
my_boxplot
dev.off()

