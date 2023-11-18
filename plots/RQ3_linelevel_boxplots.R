
library(ggplot2)

project_names <- c("BroadleafCommerce", "spring-integration", "fabric8","camel","JGroups",
                  "tomcat","nova","neutron","npm","brackets")

directory <- "/home/hareem/UofA2023/eseval_v2/eseval_timewise_batched/results_linelevel/text_metric_line_eval_result"
directory_jitline <-"/home/hareem/UofA2023/JITLine-replication-package/JITLine/text_metric_line_eval_result"

library("dplyr")
create_linelevel_plot <- function(category) {
all_data <- list()
for (project in project_names) {
  file_name <- paste0(project, "_", category, ".csv")
  file_path <- file.path(directory, file_name)
  print(file_path)
  irjit_data <- read.csv(file_path, header = FALSE)
  irjit_data$approach <- "IRJIT"
  irjit_data$metric <- category
  irjit_data$project <-project
  
  file_name_jitline <- paste0(project, "_", category, ".csv")
  file_path_jitline <- file.path(directory_jitline, file_name_jitline)
  print(file_path_jitline)
  jitline_data_temp <- read.csv(file_path_jitline, header = FALSE)
  jitline_data <- subset(jitline_data_temp, select = c('V1','V64'))
  
  colnames(jitline_data)[colnames(jitline_data) == "V64"] <- "V2"
  
  jitline_data$approach <- "JITLine"
  jitline_data$metric <- category
  jitline_data$project <- project
  
  combined_data <-rbind(irjit_data,jitline_data)
  all_data[[project]] <- combined_data
  
}
all_data <- do.call(rbind, all_data)
myplot <- ggplot(all_data, aes(x = project, y = V2, fill = as.character(approach))) +
  geom_boxplot() +
  theme( legend.position = "top", legend.title = element_blank(),
         axis.text.x=element_text(angle=20,hjust=0.5,size=8)) + 
  xlab(element_blank())+ylab(element_blank())+ scale_fill_manual(values = c( "lightblue", "lightgray","white"))+
  ylim(min(all_data$V2),max(all_data$V2))
return (myplot)
}


category =  "top_10_acc_min_df_3_300_trees"
topk_plot <-create_linelevel_plot(category)
topk_plot <- topk_plot + ylab("Top-10 accuracy")     
write_path="/home/hareem/UofA2023/eseval_v2/plots/linelevel/topk.pdf"
ggsave(topk_plot, filename=write_path, device=cairo_pdf,width=4.5, height=3.5)

category =  "recall_20_percent_effort_min_df_3_300_trees"
recall_plot <-create_linelevel_plot(category)
recall_plot <- recall_plot + ylab("Recall@20%Effort")
write_path="/home/hareem/UofA2023/eseval_v2/plots/linelevel/recall.pdf"
ggsave(recall_plot, filename=write_path, device=cairo_pdf,width=4.5, height=3.5)


category =  "effort_20_percent_recall_min_df_3_300_trees"
effort_plot <-create_linelevel_plot(category)
effort_plot <- effort_plot + ylab("Effort@20%Recall")
write_path="/home/hareem/UofA2023/eseval_v2/plots/linelevel/effort.pdf"
ggsave(effort_plot, filename=write_path, device=cairo_pdf,width=4.5, height=3.5)

category =  "IFA_min_df_3_300_trees"
IFA_plot <-create_linelevel_plot(category)
IFA_plot <- IFA_plot + ylab("IFA")
write_path="/home/hareem/UofA2023/eseval_v2/plots/linelevel/IFA.pdf"
ggsave(IFA_plot, filename=write_path, device=cairo_pdf,width=4.5, height=3.5)


library(ggpubr)
finalplot<-ggarrange(topk_plot + theme(axis.text.x =element_text(angle=20,size=8), legend.title = element_blank()) + rremove("x.title"),
                     recall_plot + theme(axis.text.x =element_text(angle=20,size=8), legend.title = element_blank())  + rremove("x.title"),
                     effort_plot  + theme(axis.text.x = element_text(angle=20,size=8),legend.title = element_blank()) + rremove("x.title"),
                     IFA_plot + theme(axis.text.x =element_text(angle=20,size=8),legend.title = element_blank())  + rremove("x.title"),
                     ncol = 2, nrow = 2, common.legend = TRUE, legend="top")

ggsave(finalplot,filename="/home/hareem/UofA2023/eseval_v2/plots/linelevel/combinedplot_linelevel.pdf", device=cairo_pdf,width=9, height=7.5)
