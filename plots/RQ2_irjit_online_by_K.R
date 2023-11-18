library(ggplot2)
#for online
df<-read.csv("/home/hareem/UofA2023/eseval_v2/eseval_timewise/all_irjit_results_on_all_commits_in_test_data.csv",header=TRUE)
dataK1 = df[df$cumulative=="Yes" & df$K==1,]
dataK3 = df[df$cumulative=="Yes" & df$K==3,]
dataK5 = df[df$cumulative=="Yes" & df$K==5,]
dataK7 = df[df$cumulative=="Yes" & df$K==7,]
dataK9 = df[df$cumulative=="Yes" & df$K==9,]
dataK11 = df[df$cumulative=="Yes" & df$K==11,]


neutron_K1 = dataK1[dataK1$project=="neutron",]
tomcat_K1 = dataK1[dataK1$project=="tomcat",]
nova_K1 =  dataK1[dataK1$project=="nova",]
broadleaf_K1 =  dataK1[dataK1$project=="BroadleafCommerce",]
fabric_K1 =  dataK1[dataK1$project=="fabric8",]
jgroups_K1 =  dataK1[dataK1$project=="JGroups",]
spring_K1 =  dataK1[dataK1$project=="spring-integration",]
camel_K1 =  dataK1[dataK1$project=="camel",]
npm_K1 = dataK1[dataK1$project=="npm",]
brackets_K1 = dataK1[dataK1$project=="brackets",]


neutron_K3 = dataK3[dataK3$project=="neutron",]
tomcat_K3 = dataK3[dataK3$project=="tomcat",]
nova_K3 =  dataK3[dataK3$project=="nova",]
broadleaf_K3 =  dataK3[dataK3$project=="BroadleafCommerce",]
fabric_K3 =  dataK3[dataK3$project=="fabric8",]
jgroups_K3 =  dataK3[dataK3$project=="JGroups",]
spring_K3 =  dataK3[dataK3$project=="spring-integration",]
camel_K3 =  dataK3[dataK3$project=="camel",]
npm_K3 = dataK3[dataK3$project=="npm",]
brackets_K3 = dataK3[dataK3$project=="brackets",]

neutron_K5 = dataK5[dataK5$project=="neutron",]
tomcat_K5 = dataK5[dataK5$project=="tomcat",]
nova_K5 =  dataK5[dataK5$project=="nova",]
broadleaf_K5 =  dataK5[dataK5$project=="BroadleafCommerce",]
fabric_K5 =  dataK5[dataK5$project=="fabric8",]
jgroups_K5 =  dataK5[dataK5$project=="JGroups",]
spring_K5 =  dataK5[dataK5$project=="spring-integration",]
camel_K5 =  dataK5[dataK5$project=="camel",]
npm_K5 = dataK5[dataK5$project=="npm",]
brackets_K5 = dataK5[dataK5$project=="brackets",]


neutron_K7 = dataK7[dataK7$project=="neutron",]
tomcat_K7 = dataK7[dataK7$project=="tomcat",]
nova_K7 =  dataK7[dataK7$project=="nova",]
broadleaf_K7 =  dataK7[dataK7$project=="BroadleafCommerce",]
fabric_K7 =  dataK7[dataK7$project=="fabric8",]
jgroups_K7 =  dataK7[dataK7$project=="JGroups",]
spring_K7 =  dataK7[dataK7$project=="spring-integration",]
camel_K7 =  dataK7[dataK7$project=="camel",]
npm_K7 = dataK7[dataK7$project=="npm",]
brackets_K7 = dataK7[dataK7$project=="brackets",]


neutron_K9 = dataK9[dataK9$project=="neutron",]
tomcat_K9 = dataK9[dataK9$project=="tomcat",]
nova_K9 =  dataK9[dataK9$project=="nova",]
broadleaf_K9 =  dataK9[dataK9$project=="BroadleafCommerce",]
fabric_K9 =  dataK9[dataK9$project=="fabric8",]
jgroups_K9 =  dataK9[dataK9$project=="JGroups",]
spring_K9 =  dataK9[dataK9$project=="spring-integration",]
camel_K9 =  dataK9[dataK9$project=="camel",]
npm_K9 = dataK9[dataK9$project=="npm",]
brackets_K9 = dataK9[dataK9$project=="brackets",]

neutron_K11 = dataK11[dataK11$project=="neutron",]
tomcat_K11 = dataK11[dataK11$project=="tomcat",]
nova_K11 =  dataK11[dataK11$project=="nova",]
broadleaf_K11 =  dataK11[dataK11$project=="BroadleafCommerce",]
fabric_K11 =  dataK11[dataK11$project=="fabric8",]
jgroups_K11 =  dataK11[dataK11$project=="JGroups",]
spring_K11 =  dataK11[dataK11$project=="spring-integration",]
camel_K11 =  dataK11[dataK11$project=="camel",]
npm_K11 = dataK11[dataK11$project=="npm",]
brackets_K11 = dataK11[dataK11$project=="brackets",]


generate_line_plot <- function(dataK1,dataK3, dataK5, dataK7, dataK9, dataK11, title) {
  plot <- ggplot() +
    geom_line(data = dataK1, aes(x = total, y = gmean, color="1"), show.legend = TRUE)+
    geom_line(data = dataK3, aes(x = total, y = gmean, color="3"), show.legend = TRUE)+
    geom_line(data = dataK5, aes(x = total, y = gmean, color="5"), show.legend = TRUE )+
    geom_line(data = dataK7, aes(x = total, y = gmean, color="7"), show.legend = TRUE)+
    geom_line(data = dataK9, aes(x = total, y = gmean, color="9"), show.legend = TRUE )+
    geom_line(data = dataK11, aes(x = total, y = gmean, color="11"), show.legend = TRUE )+
    labs(x = "timesteps", y = "G-mean", color="K") + ggtitle(title) +
    theme(axis.text.x = element_text(size= 6, hjust = 1),
          axis.text.y = element_text(angle = 45, size= 6, hjust = 1)) 
  
  return(plot)
}

neutron_plot <-   generate_line_plot(neutron_K1,neutron_K3,neutron_K5,neutron_K7,neutron_K9,neutron_K11,"Neutron")
tomcat_plot <-    generate_line_plot(tomcat_K1,tomcat_K3,tomcat_K5,tomcat_K7,tomcat_K9,tomcat_K11,"Tomcat")
nova_plot <-      generate_line_plot(nova_K1,nova_K3, nova_K5,nova_K7,nova_K9,nova_K11,"Nova")
broadleaf_plot <- generate_line_plot(broadleaf_K1,broadleaf_K3,broadleaf_K5,broadleaf_K7,broadleaf_K9,broadleaf_K11,"BroadleafCommerce")
fabric_plot <-    generate_line_plot(fabric_K1,fabric_K3,fabric_K5,fabric_K7,fabric_K9,fabric_K11,"Fabric8")
jgroups_plot <-   generate_line_plot(jgroups_K1,jgroups_K3,jgroups_K5,jgroups_K7,jgroups_K9,jgroups_K11,"JGroups")
spring_plot <-    generate_line_plot(spring_K1,spring_K3,spring_K5,spring_K7,spring_K9,spring_K11,"spring-integration")
camel_plot <-     generate_line_plot(camel_K1,camel_K3,camel_K5,camel_K7,camel_K9,camel_K11,"camel")
npm_plot <-       generate_line_plot(npm_K1,npm_K3,npm_K5,npm_K7,npm_K9,npm_K11,"npm")
brackets_plot <-  generate_line_plot(brackets_K1,brackets_K3,brackets_K5,brackets_K7,brackets_K9,brackets_K11,"brackets")

library(ggpubr)

combined_plot_all_K <- ggarrange(broadleaf_plot+rremove("xlab")+rremove("ylab"),
                           nova_plot+rremove("xlab")+rremove("ylab"),
                           neutron_plot+rremove("xlab")+rremove("ylab"),
                           tomcat_plot+rremove("xlab")+rremove("ylab"),
                           spring_plot+rremove("xlab")+rremove("ylab"),
                           fabric_plot+rremove("xlab")+rremove("ylab"),
                           camel_plot+rremove("xlab")+rremove("ylab"),
                           npm_plot+rremove("xlab")+rremove("ylab"),
                           jgroups_plot+rremove("xlab")+rremove("ylab"),
                           brackets_plot+rremove("xlab")+rremove("ylab"),
                           ncol = 2, nrow = 5,
                           common.legend = TRUE
                           )

require("grid")

final_combined_plot<-annotate_figure(combined_plot_all_K, left = textGrob("G-mean", rot = 90, vjust = 1, gp = gpar(cex = 1.0)),
                                     bottom = textGrob("timesteps", gp = gpar(cex = 1.0)))
pdf(file="/home/hareem/UofA2023/eseval_v2/plots/online/irjit_Kplot_K.pdf")
final_combined_plot
dev.off()

