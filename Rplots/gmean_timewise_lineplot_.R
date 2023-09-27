library(ggplot2)

df<-read.csv("/home/hareem/UofA2023/eseval_v2/eseval_timewise_batched/all_irjit_results_on_sampled_commits.csv",header=TRUE)
data = df[df$cumulative=="Yes" & df$K==3,]
neutron_irjit = data[data$project=="neutron",]
tomcat_irjit = data[data$project=="tomcat",]
nova_irjit =  data[data$project=="nova",]
broadleaf_irjit =  data[data$project=="BroadleafCommerce",]
fabric_irjit =  data[data$project=="fabric8",]
jgroups_irjit =  data[data$project=="JGroups",]
spring_irjit =  data[data$project=="spring-integration",]
camel_irjit =  data[data$project=="camel",]

#p <- ggplot(neutron_irjit , aes(x = total, y = gmean))
#p + geom_line() + labs(x = "Timesteps", y = "Gmean")

df2<-read.csv("/home/hareem/UofA2023/JITLine-replication-package/JITLine/all_jitline_results_on_sampled_commits.csv",header=TRUE)
data2 = df2[df2$cumulative=="Yes",]
neutron_jitline = data2[data2$project=="neutron" ,]
tomcat_jitline = data2[data2$project=="tomcat" ,]
nova_jitline = data2[data2$project=="nova" ,]
broadleaf_jitline = data2[data2$project=="BroadleafCommerce" ,]
fabric_jitline = data2[data2$project=="fabric8" ,]
jgroups_jitline = data2[data2$project=="JGroups" ,]
spring_jitline = data2[data2$project=="spring-integration" ,]
camel_jitline = data2[data2$project=="camel" ,]


generate_line_plot <- function(jitline, irjit, title) {
  plot <- ggplot() +
    geom_line(data = jitline, aes(x = total, y = gmean, color="JITLine"), show.legend = TRUE)+
    geom_line(data = irjit, aes(x = total, y = gmean, color="IRJIT"), show.legend = TRUE )+
    labs(x = "", y = "") +
    ggtitle(title) +
    theme(axis.text.x = element_text(angle = 60, size= 8, hjust = 1),legend.title = element_blank()) 

  return(plot)
}
neutron_plot <- generate_line_plot(neutron_jitline, neutron_irjit, "Neutron")
tomcat_plot <- generate_line_plot(tomcat_jitline, tomcat_irjit, "Tomcat")
nova_plot <- generate_line_plot(nova_jitline, nova_irjit, "Nova")
broadleaf_plot <- generate_line_plot(broadleaf_jitline, broadleaf_irjit, "BroadleafCommerce")
fabric_plot <- generate_line_plot(fabric_jitline, fabric_irjit, "Fabric8")
jgroups_plot <- generate_line_plot(jgroups_jitline, jgroups_irjit, "JGroups")
spring_plot <- generate_line_plot(spring_jitline, spring_irjit, "Spring-integration")
camel_plot <- generate_line_plot (camel_jitline, camel_irjit, "Camel")


library(ggpubr)

combined_plot <- ggarrange(broadleaf_plot,jgroups_plot,
                           neutron_plot,nova_plot,
                           spring_plot,tomcat_plot,
                           fabric_plot+xlab("timesteps"),
                           camel_plot+xlab("timesteps"),
                           ncol = 2, nrow = 4,
                           common.legend = TRUE)


                          
pdf(file="/home/hareem/UofA2023/eseval_v2/plots/gmean_irjit_jitline.pdf")
combined_plot
dev.off()

