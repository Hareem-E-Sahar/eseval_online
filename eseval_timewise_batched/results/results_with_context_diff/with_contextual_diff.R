df<-read.csv("/home/hareem/UofA2023/eseval_v2/eseval_timewise_batched/resultlist_neutron_K=3_with_context.csv",header=TRUE)
mean(df$FAR)
mean(df$gmean)

df2<-read.csv("/home/hareem/UofA2023/eseval_v2/eseval_timewise_batched/all_irjit_results_on_sampled_commits.csv")
df3=df2[df2$cumulative=="Yes" & df2$K==3 & df2$project=="neutron",]
mean(df3$FAR)
mean(df3$gmean)

library(ggplot2)

ggplot() +
  +     geom_line(data = df, aes(x = total, y = gmean, color="context"), show.legend = TRUE)+
  +     geom_line(data = df3, aes(x = total, y = gmean, color="without"), show.legend = TRUE )