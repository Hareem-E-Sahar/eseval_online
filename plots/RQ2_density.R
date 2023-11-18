
library(ggplot2)

irjit_data<-read.csv("/home/hareem/UofA2023/eseval_v2/eseval_timewise_batched/all_irjit_results_on_sampled_commits_by_10.csv",header=TRUE)
irjit_data = irjit_data[irjit_data$cumulative=="Yes" & irjit_data$K==3,]
jitline_data<-read.csv("/home/hareem/UofA2023/JITLine-replication-package/JITLine/all_jitline_results_on_sampled_commits_group_by10.csv",header=TRUE)
jitline_data = jitline_data[jitline_data$cumulative=="Yes",]
irjit_data <- subset(irjit_data, select = -c(K))

jitline_data$dataset <- 'JITLine'
irjit_data$dataset <- 'IRJIT'

irjit_online<-read.csv("/home/hareem/UofA2023/eseval_v2/eseval_timewise/all_irjit_results_on_all_commits_in_test_data.csv",header=TRUE)
irjit_online = irjit_online[irjit_online$cumulative=="Yes" & irjit_online$K==3,]
irjit_online$dataset <- 'IRJIT Online'
summary(irjit_online$gmean)

combined_data <- rbind(irjit_data) #jitline_data

generate_line_plot <- function() {
  plot <- ggplot(combined_data, aes(x = gmean, fill = dataset)) +
    geom_density(alpha=0.5) +
    labs(x = "G-mean", y = "Density") +
    ggtitle(title) + 
    theme(plot.title  = element_blank(),
          axis.text.x = element_text(size= 10, angle = 60, hjust = 1),
          axis.text.y = element_text(size= 10),
          legend.title = element_blank(),
          legend.position = "top")+
         scale_fill_manual(
           values = c( "IRJIT" = "lightgreen"),
                      labels = c( "IRJIT" = expression("IRJIT"["batched"]))
           )
  return(plot)
}
  
#"JITLine" = "cornflowerblue",
#"JITLine" = expression("JITLine"["batched"]),

my_plot <- generate_line_plot()

pdf(file="/home/hareem/UofA2023/eseval_v2/plots/batched/irjit_jitline_density_v2.pdf")
my_plot
dev.off()
