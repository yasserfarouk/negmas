from __future__ import annotations

import math
import sys
from functools import partial

import click

try:
    from PyQt5 import QtCore, QtGui, QtWidgets
except:
    pass
else:

    MAX_ROWS = 21

    try:
        import qdarkstyle

        _has_qdarkstyle = True
    except ModuleNotFoundError:
        _has_qdarkstyle = False

    _GTypeRole = QtCore.Qt.UserRole
    _missing = object()

    class GStyle:
        _base_style = """
            ._OptionLabel {
                font-size: 16px;
                font: bold;
                font-family: arial;
                }
            ._HelpLabel {
                font-family: arial;
                font-size: 14px;
                }
            ._InputComboBox{
                font-size: 16px;
                }
            ._InputLineEdit{
                font-size: 16px;
                }
            ._InputCheckBox{
                font-size: 16px;
                }
            ._InputSpinBox{
                font-size: 16px;
                }
            ._InputTabWidget{
                font: bold;
                font-size: 16px;
                }
            .GListView{
                font-size: 16px;
                }
            QToolTip{
                font-family: arial;
                }
            """

        def __init__(self, style=""):
            if not GStyle.check_style(style):
                self.text_color = "black"
                self.placehoder_color = "#898b8d"
                self.stylesheet = (
                    GStyle._base_style
                    + """
                        ._Spliter{
                            border: 1px inset gray;
                            }
                        """
                )
            elif style == "qdarkstyle":
                self.text_color = "#eff0f1"
                self.placehoder_color = "#898b8d"
                self.stylesheet = (
                    qdarkstyle.load_stylesheet_pyqt5()
                    + GStyle._base_style
                    + """
                        .GListView{
                            padding: 5px;
                            }
                        ._Spliter{
                            border: 5px solid gray;
                            }
                        """
                )

        @staticmethod
        def check_style(style):
            if style == "qdarkstyle":
                return _has_qdarkstyle
            return False

    _gstyle = GStyle()

    class GListView(QtWidgets.QListView):
        def __init__(self, opt):
            super().__init__()
            self.nargs = opt.nargs
            self.model = GItemModel(opt.nargs, parent=self, opt_type=opt.type, default=opt.default)
            self.setModel(self.model)
            self.delegate = GEditDelegate(self)
            self.setItemDelegate(self.delegate)
            self.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
            if self.nargs == -1:
                self.keyPressEvent = self.key_press
                self.setToolTip("'a': add a new item blow the selected one\n" "'d': delete the selected item")

        def key_press(self, e):
            if self.nargs == -1:
                if e.key() == QtCore.Qt.Key_A:
                    if len(self.selectedIndexes()) == 0:
                        self.model.insertRow(0)
                    else:
                        for i in self.selectedIndexes():
                            self.model.insertRow(i.row() + 1)
                if e.key() == QtCore.Qt.Key_D:
                    si = self.selectedIndexes()
                    for i in si:
                        self.model.removeRow(i.row())
            super().keyPressEvent(e)

    class GItemModel(QtGui.QStandardItemModel):
        def __init__(self, n, parent=None, opt_type=click.STRING, default=None):
            super(QtGui.QStandardItemModel, self).__init__(0, 1, parent)
            self.type = opt_type
            for row in range(n):
                if hasattr(default, "__len__"):
                    self.insertRow(row, default[row])
                else:
                    self.insertRow(row, default)

        def insertRow(self, idx, val=""):
            super().insertRow(idx)

            index = self.index(idx, 0, QtCore.QModelIndex())
            if val is None or val == "":
                self.setData(
                    index,
                    QtGui.QBrush(QtGui.QColor(_gstyle.placehoder_color)),
                    role=QtCore.Qt.ForegroundRole,
                )
            else:
                self.setData(index, val)

        def data(self, index, role=QtCore.Qt.DisplayRole):

            if role == QtCore.Qt.DisplayRole:
                dstr = QtGui.QStandardItemModel.data(self, index, role)
                if dstr == "" or dstr is None:
                    if isinstance(self.type, click.types.Tuple):
                        row = index.row()
                        if 0 <= row < len(self.type.types):
                            tp = self.type.types[row]
                            dstr = tp.name
                    else:
                        dstr = self.type.name
                    return dstr

            if role == _GTypeRole:
                tp = click.STRING
                if isinstance(self.type, click.types.Tuple):
                    row = index.row()
                    if 0 <= row < len(self.type.types):
                        tp = self.type.types[row]
                elif isinstance(self.type, click.types.ParamType):
                    tp = self.type
                return tp

            return QtGui.QStandardItemModel.data(self, index, role)

    class GEditDelegate(QtWidgets.QStyledItemDelegate):
        def createEditor(self, parent, option, index):
            tp = index.data(role=_GTypeRole)
            if isinstance(tp, click.Path):
                led = GLineEdit_path.from_option(tp, parent)
            else:
                led = QtWidgets.QLineEdit(parent)
            led.setPlaceholderText(tp.name)
            led.setValidator(select_type_validator(tp))
            return led

        def setEditorData(self, editor, index):
            item_var = index.data(role=QtCore.Qt.EditRole)
            if item_var is not None:
                editor.setText(str(item_var))

        def setModelData(self, editor, model, index):
            data_str = editor.text()
            if data_str == "" or data_str is None:
                model.setData(
                    index,
                    QtGui.QBrush(QtGui.QColor(_gstyle.placehoder_color)),
                    role=QtCore.Qt.ForegroundRole,
                )
            else:
                model.setData(
                    index,
                    QtGui.QBrush(QtGui.QColor(_gstyle.text_color)),
                    role=QtCore.Qt.ForegroundRole,
                )
            QtWidgets.QStyledItemDelegate.setModelData(self, editor, model, index)

    def generate_label(opt):
        show_name = getattr(opt, "show_name", _missing)
        show_name = opt.name if show_name is _missing else show_name
        param = _OptionLabel(show_name)
        param.setToolTip(getattr(opt, "help", None))
        return param

    class GStringLineEditor(click.types.StringParamType):
        def to_widget(self, opt, validator=None):
            value = _InputLineEdit()
            value.setPlaceholderText(self.name)
            if opt.default:
                value.setText(str(opt.default))
            if getattr(opt, "hide_input", False):
                value.setEchoMode(QtWidgets.QLineEdit.Password)
            value.setValidator(validator)

            def to_command():
                return [opt.opts[0], value.text()]

            return [generate_label(opt), value], to_command

    class GIntLineEditor(GStringLineEditor):
        def to_widget(self, opt):
            return GStringLineEditor.to_widget(self, opt, validator=QtGui.QIntValidator())

    class GFloatLineEditor(GStringLineEditor):
        def to_widget(self, opt):
            return GStringLineEditor.to_widget(self, opt, validator=QtGui.QDoubleValidator())

    class GFileDialog(QtWidgets.QFileDialog):
        def __init__(self, *args, exists=False, file_okay=True, dir_okay=True, **kwargs):
            super().__init__(*args, **kwargs)
            self.setOption(QtWidgets.QFileDialog.DontUseNativeDialog, True)
            self.setLabelText(QtWidgets.QFileDialog.Accept, "Select")
            if (exists, file_okay, dir_okay) == (True, True, False):
                self.setFileMode(QtWidgets.QFileDialog.ExistingFile)
            elif (exists, file_okay, dir_okay) == (False, True, False):
                self.setFileMode(QtWidgets.QFileDialog.AnyFile)
            elif (exists, file_okay, dir_okay) == (True, False, True):
                self.setFileMode(QtWidgets.QFileDialog.Directory)
            elif (exists, file_okay, dir_okay) == (False, False, True):
                self.setFileMode(QtWidgets.QFileDialog.Directory)
            elif exists == True:
                self.setFileMode(QtWidgets.QFileDialog.ExistingFile)
                self.accept = self.accept_all
            elif exists == False:
                self.setFileMode(QtWidgets.QFileDialog.AnyFile)
                self.accept = self.accept_all

        def accept_all(self):
            super().done(QtWidgets.QFileDialog.Accepted)

    class GLineEdit_path(QtWidgets.QLineEdit):
        def __init__(self, parent=None, exists=False, file_okay=True, dir_okay=True):
            super().__init__(parent)
            self.action = self.addAction(
                self.style().standardIcon(QtWidgets.QStyle.SP_DirIcon),
                QtWidgets.QLineEdit.TrailingPosition,
            )
            self.fdlg = GFileDialog(
                self,
                "Select File Dialog",
                "./",
                "*",
                exists=exists,
                file_okay=file_okay,
                dir_okay=dir_okay,
            )
            self.action.triggered.connect(self.run_dialog)

        def run_dialog(self):
            if self.fdlg.exec() == QtWidgets.QFileDialog.Accepted:
                self.setText(self.fdlg.selectedFiles()[0])

        @staticmethod
        def from_option(opt, parent=None):
            return GLineEdit_path(
                parent=parent,
                exists=opt.exists,
                file_okay=opt.file_okay,
                dir_okay=opt.dir_okay,
            )

    class GPathGLindEidt_path(click.types.Path):
        def to_widget(self, opt):
            value = GLineEdit_path(exists=self.exists, file_okay=self.file_okay, dir_okay=self.dir_okay)
            value.setPlaceholderText(self.name)
            if opt.default:
                value.setText(str(opt.default))

            def to_command():
                return [opt.opts[0], value.text()]

            return [generate_label(opt), value], to_command

    class _GLabeledSlider(QtWidgets.QSlider):
        def __init__(self, min, max, val):
            super().__init__(QtCore.Qt.Horizontal)
            self.min, self.max = min, max

            self.setMinimum(min)
            self.setMaximum(max)
            self.setValue(val)

            self.label = self.__init_label()

        def __init_label(self):
            l = max(math.ceil(math.log10(abs(x))) if x != 0 else 1 for x in [self.min, self.max])
            l += 1
            return QtWidgets.QLabel("0" * l)

    def argument_command(to_command):
        def tc():
            a = to_command()
            return a[1:]

        return tc

    class GSlider(QtWidgets.QHBoxLayout):
        def __init__(self, min=0, max=10, default=None, *args, **kwargs):
            super(QtWidgets.QHBoxLayout, self).__init__()

            self.min, self.max, self.default = min, max, default
            self.label = self.__init_label()
            self.slider = self.__init_slider()

            self.label.setText(str(self.default))

            self.addWidget(self.slider)
            self.addWidget(self.label)

        def value(self):
            return self.slider.value()

        def __init_slider(self):
            slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
            slider.setMinimum(self.min)
            slider.setMaximum(self.max)
            default_val = (self.min + self.max) // 2
            if isinstance(self.default, int):
                if self.min <= self.default <= self.max:
                    default_val = self.default
            self.default = default_val
            slider.setValue(default_val)
            slider.valueChanged.connect(lambda x: self.label.setText(str(x)))
            return slider

        def __init_label(self):
            l = max(math.ceil(math.log10(abs(x))) if x != 0 else 1 for x in [self.min, self.max])
            l += 1
            return QtWidgets.QLabel("0" * l)

    class GIntRangeGSlider(click.types.IntRange):
        def to_widget(self, opt):
            value = GSlider(min=self.min, max=self.max, default=opt.default)

            def to_command():
                return [opt.opts[0], str(value.value())]

            return [generate_label(opt), value], to_command

    class GIntRangeSlider(click.types.IntRange):
        def to_widget(self, opt):
            value = QtWidgets.QSlider(QtCore.Qt.Horizontal)
            value.setMinimum(self.min)
            value.setMaximum(self.max)

            default_val = (self.min + self.max) // 2
            if isinstance(opt.default, int):
                if self.min <= opt.default <= self.max:
                    default_val = opt.default
            value.setValue(default_val)

            def to_command():
                return [opt.opts[0], str(value.value())]

            return [generate_label(opt), value], to_command

    class GIntRangeLineEditor(click.types.IntRange):
        def to_widget(self, opt):
            value = QtWidgets.QLineEdit()
            # TODO: set validator

            def to_command():
                return [opt.opts[0], value.text()]

            return [generate_label(opt), value], to_command

    def bool_flag_option(opt):
        checkbox = _InputCheckBox(opt.name)
        if opt.default:
            checkbox.setCheckState(2)
        # set tip
        checkbox.setToolTip(opt.help)

        def to_command():
            if checkbox.checkState():
                return [opt.opts[0]]
            else:
                return opt.secondary_opts

        return [checkbox], to_command

    class GChoiceComboBox(click.types.Choice):
        def to_widget(self, opt):
            cb = _InputComboBox()
            cb.addItems(self.choices)

            def to_command():
                return [opt.opts[0], cb.currentText()]

            return [generate_label(opt), cb], to_command

    def count_option(opt):
        sb = _InputSpinBox()

        def to_command():
            return [opt.opts[0]] * int(sb.text())

        return [generate_label(opt), sb], to_command

    class GTupleGListView(click.Tuple):
        def to_widget(self, opt):
            view = GListView(opt)

            def to_command():
                _ = [opt.opts[0]]
                for idx in range(view.model.rowCount()):
                    _.append(view.model.item(idx).text())
                return _

            return [generate_label(opt), view], to_command

    def multi_text_arguement(opt):
        value = GListView(opt)

        def to_command():
            _ = []
            for idx in range(value.model.rowCount()):
                _.append(value.model.item(idx).text())
            # if opt.required and value.model.rowCount() == 0:
            # raise click.exceptions.BadParameter("Required")
            # print(opt.__dict__)
            return _

        # return [QtWidgets.QLabel(opt.name), value], to_command
        return [_OptionLabel(opt.name), value], to_command

    def select_type_validator(tp: click.types.ParamType) -> QtGui.QValidator:
        """select the right validator for `tp`"""

        if isinstance(tp, click.types.IntParamType):
            return QtGui.QIntValidator()
        elif isinstance(tp, click.types.FloatParamType):
            return QtGui.QDoubleValidator()

        return None

    def select_opt_validator(opt):
        """select the right validator for `opt`"""

        return select_type_validator(opt.type)

    def opt_to_widget(opt):
        if opt.nargs > 1:
            return GTupleGListView.to_widget(opt.type, opt)
        elif getattr(opt, "is_bool_flag", False):
            return bool_flag_option(opt)
        elif getattr(opt, "count", False):
            return count_option(opt)
        elif isinstance(opt.type, click.types.Choice):
            return GChoiceComboBox.to_widget(opt.type, opt)
        elif isinstance(opt.type, click.types.Path):
            return GPathGLindEidt_path.to_widget(opt.type, opt)
        elif isinstance(opt.type, click.types.IntRange):
            return GIntRangeGSlider.to_widget(opt.type, opt)
        elif isinstance(opt.type, click.types.IntParamType):
            return GIntLineEditor.to_widget(opt.type, opt)
        elif isinstance(opt.type, click.types.FloatParamType):
            return GFloatLineEditor.to_widget(opt.type, opt)
        else:
            return GStringLineEditor.to_widget(opt.type, opt)

    def _to_widget(opt):
        # customed widget

        if isinstance(opt.type, click.types.FuncParamType):
            if hasattr(opt.type.func, "to_widget"):
                return opt.type.func.to_widget()
        elif hasattr(opt.type, "to_widget"):
            return opt.type.to_widget()

        if isinstance(opt, click.core.Argument):
            if opt.nargs == 1:
                w, tc = opt_to_widget(opt)

                return w, argument_command(tc)
            elif opt.nargs > 1 or opt.nargs == -1:
                return multi_text_arguement(opt)
        else:
            return opt_to_widget(opt)

    def layout_append_opts(layout, opts):
        params_func = []
        widgets = []
        i = 0

        for i, para in enumerate(opts):
            widget, value_func = _to_widget(para)
            widgets.append(widget)
            params_func.append(value_func)

            for idx, w in enumerate(widget):

                if isinstance(w, QtWidgets.QLayout):
                    layout.addLayout(w, i, idx)
                else:
                    layout.addWidget(w, i, idx)

        return layout, params_func, widgets

    def generate_sysargv(cmd_list):
        argv_list = []

        for name, func_list in cmd_list:
            argv_list.append(name)

            for value_func in func_list:
                argv_list += value_func()

        return argv_list

    class _Spliter(QtWidgets.QFrame):
        def __init__(self, parent=None):
            super().__init__(parent=parent)
            self.setFrameShape(QtWidgets.QFrame.HLine)

    class _InputComboBox(QtWidgets.QComboBox):
        pass

    class _InputTabWidget(QtWidgets.QTabWidget):
        pass

    class _HelpLabel(QtWidgets.QLabel):
        pass

    class _OptionLabel(QtWidgets.QLabel):
        pass

    class _InputLineEdit(QtWidgets.QLineEdit):
        pass

    class _InputCheckBox(QtWidgets.QCheckBox):
        pass

    class _InputSpinBox(QtWidgets.QSpinBox):
        pass

    class CommandLayout(QtWidgets.QGridLayout):
        def __init__(self, func, run_exit, parent_layout=None):
            super().__init__()
            self.parent_layout = parent_layout
            self.func = func
            self.run_exit = run_exit

            if func.help:
                label = _HelpLabel(func.help)
                label.setWordWrap(True)
                self.addWidget(label, 0, 0, 1, 2)
                frame = _Spliter()
                self.addWidget(frame, 1, 0, 1, 4)
            self.params_func, self.widgets = self.append_opts(self.func.params)

        def add_sysargv(self):
            if hasattr(self.parent_layout, "add_sysargv"):
                self.parent_layout.add_sysargv()
            sys.argv += generate_sysargv([(self.func.name, self.params_func)])

        def append_opts(self, opts):
            params_func = []
            widgets = []

            def _nwidgets(x):
                return -len(_to_widget(x)[0])

            opts = sorted(opts, key=_nwidgets)
            row, col = 0, 0
            for para in opts:
                widget, value_func = _to_widget(para)
                widgets.append(widget)
                params_func.append(value_func)

                for idx, w in enumerate(widget):
                    if isinstance(w, QtWidgets.QLayout):
                        self.addLayout(w, row + 2, 2 * col + idx)
                    else:
                        self.addWidget(w, row + 2, 2 * col + idx)
                self.setRowStretch(row, 5)
                row += 1
                if row >= MAX_ROWS - 1:
                    row = 0
                    col += 1

            return params_func, widgets

        def generate_cmd_button(self, label, cmd_slot, tooltip=""):
            button = QtWidgets.QPushButton(label)
            button.setToolTip(tooltip)
            button.clicked.connect(self.clean_sysargv)
            button.clicked.connect(self.add_sysargv)
            button.clicked.connect(cmd_slot)

            return button

        def add_cmd_button(self, label, cmd_slot, pos=None):
            run_button = self.generate_cmd_button(label, cmd_slot)

            if pos is None:
                # pos = self.rowCount() + 1, 0
                pos = 0, 2
            self.addWidget(run_button, pos[0], pos[1])

        def add_cmd_buttons(self, args):
            # row = self.rowCount() + 1
            row = 0
            cmd_layout = QtWidgets.QGridLayout()
            cmd_layout.setHorizontalSpacing(20)

            for col, arg in enumerate(args):
                button = self.generate_cmd_button(**arg)
                cmd_layout.addWidget(button, col, 0)
            self.addLayout(cmd_layout, row, 2, 1, 2)

        @QtCore.pyqtSlot()
        def clean_sysargv(self):
            sys.argv = []

    class RunCommand(QtCore.QRunnable):
        def __init__(self, func, run_exit):
            super().__init__()
            self.func = func
            self.run_exit = run_exit

        @QtCore.pyqtSlot()
        def run(self):
            print(sys.argv)
            try:
                self.func(standalone_mode=self.run_exit)
            except click.exceptions.BadParameter as bpe:
                # warning message
                msg = QtWidgets.QMessageBox()
                msg.setIcon(QtWidgets.QMessageBox.Warning)
                msg.setText(bpe.format_message())
                msg.exec_()
            except Exception as bpe:
                msg = QtWidgets.QMessageBox()
                msg.setIcon(QtWidgets.QMessageBox.Warning)
                msg.setText(repr(bpe))
                msg.exec_()
            # if self.outputEdit is not None:
            # self.outputEdit.show()

    class GCommand(click.Command):
        def __init__(self, new_thread=True, *arg, **args):
            super().__init__(*arg, **args)
            self.new_thread = new_thread

    class GOption(click.Option):
        def __init__(self, *arg, show_name=_missing, **args):
            super().__init__(*arg, **args)
            self.show_name = show_name

    # def normalOutputWritten(t):
    # """Append text to the QTextEdit."""
    # Maybe QTextEdit.append() works as well, but this is how I do it:
    # cursor = text.textCursor()
    # cursor.movePosition(QtGui.QTextCursor.End)
    # cursor.insertText(t)
    # text.setTextCursor(cursor)
    # text.ensureCursorVisible()

    class GuiStream(QtCore.QObject):
        textWritten = QtCore.pyqtSignal(str)

        def flush(self):
            pass

        def write(self, text):
            self.textWritten.emit(str(text))

    class OutputEdit(QtWidgets.QTextEdit):
        def print(self, text):
            cursor = self.textCursor()
            cursor.movePosition(QtGui.QTextCursor.End)
            cursor.insertText(text)
            self.setTextCursor(cursor)
            self.ensureCursorVisible()

    class App(QtWidgets.QWidget):
        def __init__(
            self,
            func,
            run_exit,
            new_thread,
            output="gui",
            left=10,
            top=10,
            width=800,
            height=280,
        ):
            """
            Parameters
            ----------
            output : str
                'gui': [default] redirect screen output to the gui
                'term': do nothing
            """
            super().__init__()
            self.new_thread = new_thread
            self.title = func.name
            self.func = func
            self.initUI(run_exit, QtCore.QRect(left, top, width, height))
            self.threadpool = QtCore.QThreadPool()
            self.outputEdit = self.initOutput(output)

        def initOutput(self, output):
            if output == "gui":
                sys.stdout = GuiStream()
                sys.stderr = sys.stdout
                text = OutputEdit()
                text.setReadOnly(True)
                sys.stdout.textWritten.connect(text.print)
                sys.stdout.textWritten.connect(text.show)
                # sys.stdout.textWritten.connect(text.append)
                # text.show()

                return text
            else:
                return None

        def initCommandUI(self, func, run_exit, parent_layout=None):
            opt_set = CommandLayout(func, run_exit, parent_layout=parent_layout)

            if isinstance(func, click.MultiCommand):
                tabs = _InputTabWidget()

                for cmd, f in func.commands.items():
                    sub_opt_set = self.initCommandUI(f, run_exit, parent_layout=opt_set)
                    tab = QtWidgets.QWidget()
                    tab.setLayout(sub_opt_set)
                    tabs.addTab(tab, cmd)
                opt_set.addWidget(tabs, opt_set.rowCount(), 0, 1, 4)
                # return opt_set
            elif isinstance(func, click.Command):
                new_thread = getattr(func, "new_thread", self.new_thread)
                opt_set.add_cmd_buttons(
                    args=[
                        {
                            "label": "&Run",
                            "cmd_slot": partial(self.run_cmd, new_thread=new_thread),
                            "tooltip": "run command",
                        },
                        {
                            "label": "&Copy",
                            "cmd_slot": self.copy_cmd,
                            "tooltip": "copy command to clipboard",
                        },
                    ]
                )

            return opt_set

        def initUI(self, run_exit, geometry):
            self.run_exit = run_exit
            self.setWindowTitle(self.title)
            # self.setGeometry(self.left, self.top, self.width, self.height)
            self.setGeometry(geometry)
            self.opt_set = self.initCommandUI(
                self.func,
                run_exit,
            )
            self.setLayout(self.opt_set)
            self.show()

        @QtCore.pyqtSlot()
        def copy_cmd(self):
            cb = QtWidgets.QApplication.clipboard()
            cb.clear(mode=cb.Clipboard)
            cmd_text = " ".join(sys.argv)
            cb.setText(cmd_text, mode=cb.Clipboard)

            msg = QtWidgets.QMessageBox()
            msg.setIcon(QtWidgets.QMessageBox.Information)
            msg.setText(f"copy '{cmd_text}' to clipboard")
            msg.exec_()

        def run_cmd(self, new_thread):
            runcmd = RunCommand(self.func, self.run_exit)

            if new_thread:
                self.threadpool.start(runcmd)
            else:
                runcmd.run()

    def gui_it(click_func, style="qdarkstyle", **argvs) -> None:
        """
        Parameters
        ----------
        click_func
        `new_thread` is used for qt-based func, like matplotlib
        """
        global _gstyle
        _gstyle = GStyle(style)
        app = QtWidgets.QApplication(sys.argv)
        app.setStyleSheet(_gstyle.stylesheet)

        # set the default value for argvs
        argvs["run_exit"] = argvs.get("run_exit", False)
        argvs["new_thread"] = argvs.get("new_thread", False)

        _ = App(click_func, **argvs)
        sys.exit(app.exec_())

    def gui_option(f: click.core.BaseCommand) -> click.core.BaseCommand:
        """decorator for adding '--gui' option to command"""
        # TODO: add run_exit, new_thread
        def run_gui_it(ctx, param, value):
            if not value or ctx.resilient_parsing:
                return
            f.params = [p for p in f.params if not p.name == "gui"]
            gui_it(f)
            ctx.exit()

        return click.option(
            "--gui",
            is_flag=True,
            callback=run_gui_it,
            help="run with gui",
            expose_value=False,
            is_eager=False,
        )(f)
