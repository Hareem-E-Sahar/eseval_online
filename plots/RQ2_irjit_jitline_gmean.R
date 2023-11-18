library(ggplot2)
df<-read.csv("/home/hareem/UofA2023/eseval_v2/eseval_timewise_batched/all_irjit_results_on_sampled_commits_by_10.csv",header=TRUE)
data3 = df[df$cumulative=="Yes" & df$K==3,]
#write.csv(data3,"/home/hareem/UofA2023/eseval_v2/eseval_timewise_batched/irjit_for_chatgpt.csv")
neutron_irjit3 = data3[data3$project=="neutron",]
tomcat_irjit3 = data3[data3$project=="tomcat",]
broadleaf_irjit3 =  data3[data3$project=="BroadleafCommerce",]
jgroups_irjit3 =  data3[data3$project=="JGroups",]
camel_irjit3 =  data3[data3$project=="camel",]
spring_irjit3 =  data3[data3$project=="spring-integration",]
nova_irjit3 =  data3[data3$project=="nova",]
fabric_irjit3 =  data3[data3$project=="fabric8",]
npm_irjit3    = data3[data3$project=="npm",]
brackets_irjit3 = data3[data3$project=="brackets",]
  

df2<-read.csv("/home/hareem/UofA2023/JITLine-replication-package/JITLine/all_jitline_results_on_sampled_commits_group_by10.csv",header=TRUE)
data2 = df2[df2$cumulative=="Yes",]
#write.csv(data3,"/home/hareem/UofA2023/JITLine-replication-package/JITLine/jitline_for_chatgpt.csv")

neutron_jitline = data2[data2$project=="neutron" ,]
tomcat_jitline = data2[data2$project=="tomcat" ,]
nova_jitline = data2[data2$project=="nova" ,]
broadleaf_jitline = data2[data2$project=="BroadleafCommerce" ,]
fabric_jitline = data2[data2$project=="fabric8" ,]
jgroups_jitline = data2[data2$project=="JGroups" ,]
spring_jitline = data2[data2$project=="spring-integration" ,]
camel_jitline = data2[data2$project=="camel" ,]
npm_jitline = data2[data2$project=="npm",]
brackets_jitline = data2[data2$project=="brackets",]


generate_line_plot <- function(jitline, irjit3, title) {
  irjit3 <- subset(irjit3, select = -c(K))
  jitline$approach <- 'JITLine'
  irjit3$approach <- 'IRJIT'
  combined_data <- rbind(jitline, irjit3)
  plot <- ggplot(combined_data, aes(x = total, y=gmean, color = approach)) +
    geom_line( show.legend = TRUE)+
    #geom_line(combined_data, aes(x = total, y = gmean, fill="approach"), show.legend = TRUE)+
    labs(x = "", y = "") +
    ggtitle(title) +
    theme(plot.title  = element_text(size = 8),
          axis.text.x = element_text(size= 8, angle = 60, hjust = 1),
          axis.text.y = element_text(size= 8),
          legend.title = element_blank())+
    scale_color_manual(values=c("JITLine"="lightblue4", "IRJIT"="orange"),
                      labels=c("JITLine"=expression("JITLine"["batched"]), "IRJIT"=expression("IRJIT"["batched"])))   
  return(plot)
}
neutron_plot <- generate_line_plot(neutron_jitline, neutron_irjit3, "Neutron")
tomcat_plot <- generate_line_plot(tomcat_jitline, tomcat_irjit3, "Tomcat")
nova_plot <- generate_line_plot(nova_jitline, nova_irjit3, "Nova")
broadleaf_plot <- generate_line_plot(broadleaf_jitline,  broadleaf_irjit3, "BroadleafCommerce")
fabric_plot <- generate_line_plot(fabric_jitline,  fabric_irjit3, "Fabric8")
jgroups_plot <- generate_line_plot(jgroups_jitline,  jgroups_irjit3, "JGroups")
spring_plot <- generate_line_plot(spring_jitline, spring_irjit3, "Spring-integration")
camel_plot <- generate_line_plot (camel_jitline, camel_irjit3, "Camel")
npm_plot <- generate_line_plot (npm_jitline, npm_irjit3, "Npm")
brackets_plot <- generate_line_plot (brackets_jitline, brackets_irjit3, "Brackets")


library(ggpubr)

combined_plot <- ggarrange(broadleaf_plot+theme(axis.text.x=element_blank(), axis.ticks.x=element_blank()),
                           tomcat_plot+theme(axis.text.x=element_blank(), axis.ticks.x=element_blank()),
                           jgroups_plot+theme(axis.text.x=element_blank(), axis.ticks.x=element_blank()),
                           nova_plot+theme(axis.text.x=element_blank(), axis.ticks.x=element_blank()),
                           spring_plot+theme(axis.text.x=element_blank(), axis.ticks.x=element_blank()),
                           neutron_plot+theme(axis.text.x=element_blank(), axis.ticks.x=element_blank()),
                           npm_plot+theme(axis.text.x=element_blank(), axis.ticks.x=element_blank()),
                           brackets_plot+theme(axis.text.x=element_blank(), axis.ticks.x=element_blank()),
                           fabric_plot,
                           camel_plot,
                           ncol = 2, nrow = 5,
                           common.legend = TRUE)

library(grid)

final_combined_plot<-annotate_figure(combined_plot, left = textGrob("G-mean", rot = 90, vjust = 1, gp = gpar(cex = 1.0)),
                                     bottom = textGrob("timesteps", gp = gpar(cex = 1.0)))

pdf(file="/home/hareem/UofA2023/eseval_v2/plots/batched/irjit_jitline_by10.pdf",width=8.5,height=10)
final_combined_plot
dev.off()
