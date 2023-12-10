# Load libraries
library(readr)
library(dplyr)
library(pROC)

# Set options
options(scipen = 99)
options(digits = 4)

# Read the data from the CSV file and store it in the variable df
#df <- read.csv("/Users/vamsisaitumu/Desktop/code/hotel_reservations.csv")
setwd("~/Documents/Jyoti MSBA/Fall'23/Predictive Analytics_BUAN 6337/Project/Final")
df <- read.csv("hotel_reservations.csv")

df$market_segment_type <- ifelse(df$market_segment_type == 'Complementary', "Other", df$market_segment_type)
df$market_segment_type <- ifelse(df$market_segment_type == 'Corporate', 'Other', df$market_segment_type)
df$market_segment_type <- ifelse(df$market_segment_type == 'Aviation', "Other", df$market_segment_type)

unique(df$market_segment_type)

# Converting type_of_meal_plan, room_type_reserved, market_segment_type as factor
df$type_of_meal_plan <- as.factor(df$type_of_meal_plan)
df$room_type_reserved <- as.factor(df$room_type_reserved)
df$market_segment_type <- as.factor(df$market_segment_type)

# Converting booking_status to 0/1
df$booking_status <- ifelse(df$booking_status == 'Canceled', 1, 0)




# Getting weekday from arrival_year, arrival_month, and arrival_day
df$arrival_date_calc = paste(df$arrival_year, df$arrival_month, df$arrival_date, sep = '-')
df$arrival_date_calc <- ifelse(df$arrival_date_calc == '2018-2-29', '2018-3-1', df$arrival_date_calc)
df$arrival_day <- weekdays(as.Date(df$arrival_date_calc))

# Converting arrival_day as Factor with proper leveling
df$arrival_day <- factor(df$arrival_day, levels=c("Monday", "Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"))

# Creating a new feature - previous_cancellation_ratio
df$prev_bookings <- df$no_of_previous_cancellations + df$no_of_previous_bookings_not_canceled
df$prev_cancellation_bucket <- ifelse(df$prev_bookings < 2, 'LessData' , ifelse(df$no_of_previous_cancellations/df$prev_bookings > 0.3, 'MoreCan', 'LessCan'))

write.csv(df, file = "~/Documents/Jyoti MSBA/Fall'23/Predictive Analytics_BUAN 6337/Project/Final/hotel.csv", row.names = FALSE)

# Display table for prev_cancellation_bucket
table(df$prev_cancellation_bucket)

# Display summary of the data
str(df)

# Check NULL values
sapply(df, function(x) sum(is.na(x)))

# Create a new Data frame - new_df
# Dropping Booking_ID, arrival_year, arrival_month, and arrival_day
# Dropping prev_bookings, no_of_previous_cancellations, no_of_previous_bookings_not_canceled
new_df <- subset(df, select = -c(Booking_ID, arrival_year, arrival_month, arrival_date, arrival_date_calc, prev_bookings, no_of_previous_cancellations, no_of_previous_bookings_not_canceled))

# Splitting data into training and validation sets
set.seed(123)
train_indices <- sample(seq_len(nrow(new_df)), size = 0.7 * nrow(new_df))
train_data <- new_df[train_indices, ]
valid_data <- new_df[-train_indices, ]

# Logistic Regression
train_data$arrival_day = relevel(train_data$arrival_day, ref=4)
model <- glm(booking_status ~ ., data = train_data, family = "binomial")
summary(model)

# Odds Ratio

odds_ratios <- round(exp(coef(model)),4)
pvalues <- round(coef(summary(model))[, 'Pr(>|z|)'],4)


# c - Create a data frame with odds ratios and p-values
df_odds_pval <- data.frame(OR = odds_ratios, p_value = pvalues)
df_odds_pval

df_odds_pval$VarName <- rownames(df_odds_pval)
write.csv(df_odds_pval, "OR.csv", row.names = FALSE)

# Predictions on the validation set
predictions <- predict(model, newdata = valid_data, type = "response")

# Evaluate model performance
roc_curve <- roc(valid_data$booking_status, predictions)
auc_value <- auc(roc_curve)
print(paste("AUC-ROC:", auc_value))
plot(roc_curve, main = "ROC Curve", col = "blue", lwd = 2)

# Confusion matrix
conf_matrix <- table(predicted = round(predictions), actual = valid_data$booking_status)
print(conf_matrix)


# Save confusion matrix to CSV
write.csv(conf_matrix, file = "/Users/vamsisaitumu/Desktop/code/conf_matrix.csv", row.names = TRUE)

# Calculate accuracy, precision, and recall
accuracy <- sum(diag(conf_matrix)) / sum(conf_matrix)
precision <- conf_matrix[2, 2] / sum(conf_matrix[, 2])
recall <- conf_matrix[2, 2] / sum(conf_matrix[2, ])
print(paste("Accuracy:", accuracy))
print(paste("Precision:", precision))
print(paste("Recall:", recall))
# Save metrics to CSV
metrics <- data.frame(Metric = c("Accuracy", "Precision", "Recall"),
                      Value = c(accuracy, precision, recall))

write.csv(metrics, file = "/Users/vamsisaitumu/Desktop/code/metrics.csv", row.names = FALSE)




# Create a new data frame for actual and predicted classes with cancellation status
class_df <- data.frame(
  Actual_Class = ifelse(valid_data$booking_status == 1, "Canceled", "Not Canceled"),
  Predicted_Class = ifelse(round(predictions) == 1, "Canceled", "Not Canceled")
)

# Display the first few rows of the data frame
head(class_df)

# Save the data frame to CSV
write.csv(class_df, file = "/Users/vamsisaitumu/Desktop/code/actual_vs_predicted_classes_with_cancellation.csv", row.names = FALSE)
