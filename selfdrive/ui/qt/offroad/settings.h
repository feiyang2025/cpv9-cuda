#pragma once

#include <map>
#include <string>

#include <QButtonGroup>
#include <QFrame>
#include <QLabel>
#include <QPushButton>
#include <QStackedWidget>
#include <QWidget>

#include "selfdrive/ui/ui.h"
#include "selfdrive/ui/qt/util.h"
#include "selfdrive/ui/qt/widgets/controls.h"

// ********** settings window + top-level panels **********
class SettingsWindow : public QFrame {
  Q_OBJECT

public:
  explicit SettingsWindow(QWidget *parent = 0);
  void setCurrentPanel(int index, const QString &param = "");

protected:
  void showEvent(QShowEvent *event) override;

signals:
  void closeSettings();
  void reviewTrainingGuide();
  void showDriverView();
  void expandToggleDescription(const QString &param);

private:
  QPushButton *sidebar_alert_widget;
  QWidget *sidebar_widget;
  QButtonGroup *nav_btns;
  QStackedWidget *panel_widget;
};

class DevicePanel : public ListWidget {
  Q_OBJECT
public:
  explicit DevicePanel(SettingsWindow *parent);

signals:
  void reviewTrainingGuide();
  void showDriverView();

private slots:
  void poweroff();
  void reboot();
  //re_Calibration
  void calibration();
  void updateCalibDescription();

private:
  Params params;
  ButtonControl *pair_device;
};

class TogglesPanel : public ListWidget {
  Q_OBJECT
public:
  explicit TogglesPanel(SettingsWindow *parent);
  void showEvent(QShowEvent *event) override;

public slots:
  void expandToggleDescription(const QString &param);

private slots:
  void updateState(const UIState &s);

private:
  Params params;
  std::map<std::string, ParamControl*> toggles;
  ButtonParamControl *long_personality_setting;

  void updateToggles();
};

class SoftwarePanel : public ListWidget {
  Q_OBJECT
public:
  explicit SoftwarePanel(QWidget* parent = nullptr);

private:
  void showEvent(QShowEvent *event) override;
  void updateLabels();
  void checkForUpdates();

  bool is_onroad = false;

  QLabel *onroadLbl;
  LabelControl *versionLbl;
  ButtonControl *installBtn;
  ButtonControl *downloadBtn;
  ButtonControl *onOffRoadBtn;
  ButtonControl *targetBranchBtn;

  Params params;
  ParamWatcher *fs_watch;
};

// Forward declaration
class FirehosePanel;

class CarrotPanel : public QWidget {
  Q_OBJECT

private:
  QStackedLayout* main_layout = nullptr;
  QWidget* homeScreen = nullptr;
  int currentCarrotIndex = 0;

  QWidget* homeWidget;
  QVBoxLayout* carrotLayout;

  ListWidget* cruiseToggles;
  ListWidget* latLongToggles;
  ListWidget* pathToggles;
  ListWidget* dispToggles;
  ListWidget* startToggles;
  ListWidget* speedToggles;
  ListWidget* navToggles;

  void togglesCarrot(int widgetIndex);
  void updateButtonStyles();

public:
  explicit CarrotPanel(QWidget* parent = nullptr);
};

class CValueControl : public AbstractControl {
  Q_OBJECT

public:
  // map: 可选的 数值->中文 标签表（如 {"0":"自学习","1":"自定义","2":"默认值"}）。
  // 传了 map 后 label 显示中文, 按钮行为变为"模式切换"(在 min..max 间循环)。
  // map 下标对应 m_min 起始的值（如 modelid 从 -1 起, 下标0=-1）。
  // map 非空时点击 label 弹出大按钮选择框（车机友好, 替代 +/- 循环）。
  CValueControl(const QString& params, const QString& title, const QString& desc,
                int min, int max, int unit = 1, const QStringList& map = {});

private slots:
  void increaseValue();
  void decreaseValue();

private:
  void showEvent(QShowEvent* event) override;
  void refresh();
  void adjustValue(int delta);
  void showPopup();
  void mouseReleaseEvent(QMouseEvent* event) override;

  QStringList m_map;  // 空 = 普通数值显示; 非空 = 中文模式名显示 + 循环切换 + 点击弹窗

  QPushButton btnplus;
  QPushButton btnminus;
  ElidedLabel label;

  QString m_params;
  int m_min;
  int m_max;
  int m_unit;
};

// 字符串枚举选择控件: 底层参数是字符串(如 Model="Classic"/"BigCombo"),
// 选项列表由构造传入(可动态扫描), 整行点击弹大按钮选择框。
class CStringEnumControl : public AbstractControl {
  Q_OBJECT

public:
  CStringEnumControl(const QString& params, const QString& title, const QString& desc,
                     const QStringList& options, const QStringList& labels = {},
                     QWidget* parent = nullptr);

private:
  void showEvent(QShowEvent* event) override;
  void refresh();
  void showPopup();
  void mouseReleaseEvent(QMouseEvent* event) override;

  QStringList m_options;  // 实际写入的字符串值
  QStringList m_labels;   // 显示名(空 = 用值本身)
  QString m_params;
  ElidedLabel label;
};
