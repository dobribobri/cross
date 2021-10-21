# -*- coding: utf-8 -*-
import PIL.Image
from PIL import ImageTk
from tkinter import filedialog
from tkinter import *
from cross import *
from cross import CommonOperations as co
import warnings
from matplotlib import pyplot as plt
import matplotlib as mpl
mpl.rcParams.update({'font.size': 5, 'axes.linewidth': 0.5,
                     'xtick.major.size': 2, 'ytick.major.size': 2,
                     'xtick.major.width': 0.4, 'ytick.major.width': 0.4})


class ImageField(Frame):
    def __init__(self, master, width: int, height: int, image_path: str = None):
        Frame.__init__(self, master=master)
        self.x = self.y = 0
        self.canvas = Canvas(self, cursor="cross", width=width, height=height)

        self.sbarv = Scrollbar(self, orient=VERTICAL)
        self.sbarh = Scrollbar(self, orient=HORIZONTAL)
        self.sbarv.config(command=self.canvas.yview)
        self.sbarh.config(command=self.canvas.xview)

        self.canvas.config(yscrollcommand=self.sbarv.set)
        self.canvas.config(xscrollcommand=self.sbarh.set)

        self.canvas.grid(row=0, column=0, sticky=N + S + E + W)
        self.sbarv.grid(row=0, column=1, stick=N + S)
        self.sbarh.grid(row=1, column=0, sticky=E + W)

        self.canvas.bind("<ButtonPress-1>", self.on_button_press)
        self.canvas.bind("<B1-Motion>", self.on_move_press)
        self.canvas.bind("<ButtonRelease-1>", self.on_button_release)

        self.rect = None

        self.startX = None
        self.startY = None
        self.curX = None
        self.curY = None

        self.width, self.height = width, height
        self.im, self.tk_im = None, None
        self.w_im, self.h_im = None, None
        self.resize_coeff, self.w_im_resized, self.h_im_resized = None, None, None
        self.image_on_canvas = None

        if image_path:
            self.reload_image(image_path)

        self.imf_conn = None

        self.boxed = False

    def reload_image(self, image_path: str) -> None:
        self.im = PIL.Image.open(image_path)
        self.w_im, self.h_im = self.im.size
        self.resize_coeff = self.width / self.w_im
        self.h_im_resized = int(self.resize_coeff * self.h_im)
        self.w_im_resized = int(self.resize_coeff * self.w_im)
        self.im = self.im.resize((self.w_im_resized, self.h_im_resized))
        self.canvas.config(scrollregion=(0, 0, self.w_im_resized, self.h_im_resized))
        self.tk_im = ImageTk.PhotoImage(self.im)
        if not self.image_on_canvas:
            self.image_on_canvas = self.canvas.create_image(0, 0, anchor="nw", image=self.tk_im)
        else:
            self.canvas.itemconfig(self.image_on_canvas, image=self.tk_im)

    def on_button_press(self, event):
        self.startX = self.canvas.canvasx(event.x)
        self.startY = self.canvas.canvasy(event.y)
        if not self.rect:
            self.rect = self.canvas.create_rectangle(self.x, self.y, 2, 2, outline='white')
        if self.imf_conn:
            self.imf_conn.startX = self.startX
            self.imf_conn.startY = self.startY
            if not self.imf_conn.rect:
                self.imf_conn.rect = \
                    self.imf_conn.canvas.create_rectangle(self.imf_conn.x, self.imf_conn.y, 2, 2, outline='white')

    def on_move_press(self, event):
        self.curX = self.canvas.canvasx(event.x)
        self.curY = self.canvas.canvasy(event.y)
        if self.imf_conn:
            self.imf_conn.curX = self.canvas.canvasx(event.x)
            self.imf_conn.curY = self.canvas.canvasy(event.y)
        w, h = self.canvas.winfo_width(), self.canvas.winfo_height()
        if event.x > 0.9 * w:
            self.canvas.xview_scroll(1, 'units')
        elif event.x < 0.1 * w:
            self.canvas.xview_scroll(-1, 'units')
        if event.y > 0.9 * h:
            self.canvas.yview_scroll(1, 'units')
        elif event.y < 0.1 * h:
            self.canvas.yview_scroll(-1, 'units')
        if self.imf_conn:
            w, h = self.imf_conn.canvas.winfo_width(), self.imf_conn.canvas.winfo_height()
            if event.x > 0.9 * w:
                self.imf_conn.canvas.xview_scroll(1, 'units')
            elif event.x < 0.1 * w:
                self.imf_conn.canvas.xview_scroll(-1, 'units')
            if event.y > 0.9 * h:
                self.imf_conn.canvas.yview_scroll(1, 'units')
            elif event.y < 0.1 * h:
                self.imf_conn.canvas.yview_scroll(-1, 'units')
        if self.curX > self.w_im_resized:
            self.curX = self.w_im_resized
        if self.curY > self.h_im_resized:
            self.curY = self.h_im_resized
        if self.curX < 0:
            self.curX = self.startX
            self.startX = 0
        if self.curY < 0:
            self.curY = self.startY
            self.startY = 0
        if self.imf_conn:
            if self.imf_conn.curX > self.imf_conn.w_im_resized:
                self.imf_conn.curX = self.imf_conn.w_im_resized
            if self.imf_conn.curY > self.imf_conn.h_im_resized:
                self.imf_conn.curY = self.imf_conn.h_im_resized
            if self.imf_conn.curX < 0:
                self.imf_conn.curX = self.imf_conn.startX
                self.imf_conn.startX = 0
            if self.imf_conn.curY < 0:
                self.imf_conn.curY = self.imf_conn.startY
                self.imf_conn.startY = 0
        if self.boxed:
            self.curY = self.startY + self.curX - self.startX
        self.canvas.coords(self.rect, self.startX, self.startY, self.curX, self.curY)
        if self.imf_conn:
            if self.imf_conn.boxed:
                self.imf_conn.curY = self.imf_conn.startY + self.imf_conn.curX - self.imf_conn.startX
            self.imf_conn.canvas.coords(self.imf_conn.rect, self.imf_conn.startX, self.imf_conn.startY,
                                        self.imf_conn.curX, self.imf_conn.curY)

    def on_button_release(self, _event):
        if self.startX > self.curX:
            self.startX, self.curX = self.curX, self.startX
        if self.startY > self.curY:
            self.startY, self.curY = self.curY, self.startY
        if self.imf_conn:
            if self.imf_conn.startX > self.imf_conn.curX:
                self.imf_conn.startX, self.imf_conn.curX = self.imf_conn.curX, self.imf_conn.startX
            if self.imf_conn.startY > self.imf_conn.curY:
                self.imf_conn.startY, self.imf_conn.curY = self.imf_conn.curY, self.imf_conn.startY

    def connect(self, imf: 'ImageField') -> None:
        self.imf_conn = imf

    def get_selected_area_coords(self) -> Union[Tuple[Tuple[int, int], Tuple[int, int]], bool]:
        try:
            x0 = int(1 / self.resize_coeff * self.startX)
            y0 = int(1 / self.resize_coeff * self.startY)
            x1 = int(1 / self.resize_coeff * self.curX) - 1
            y1 = int(1 / self.resize_coeff * self.curY) - 1
        except TypeError:
            print('Выделите прямоугольную область')
            return False

        if x0 == 0:
            x0 += 3
        if y0 == 0:
            y0 += 3
        if x1 == self.w_im - 1:
            x1 -= 3
        if y1 == self.h_im - 1:
            y1 -= 3
        return (x0, y0), (x1, y1)


imf_width = 300
imf_height = 430


class View:
    def __init__(self,
                 _imgs: Union[Tuple[RadarImage, RadarImage], List[RadarImage]],
                 _imfs: Union[Tuple[ImageField, ImageField], List[ImageField]],
                 _mode: DType = DType.AMPLITUDES, _scale=True, _elim=True,
                 _smoothing: int = 0, _exclusive=False, _order: int = 1,
                 _plot_x=True, _plot_y=False, _clear_plots=False,
                 _plot_mode: DType = DType.AMPLITUDES, _pcs=None,
                 _correlation_mode: CorrelationMode = CorrelationMode.MODE1,
                 _boxed=False, _show_legend=True):
        self.imgs = _imgs
        self.imfs = _imfs
        self.mode, self.scale, self.elim = _mode, _scale, _elim
        self.smoothing, self.exclusive, self.order = _smoothing, _exclusive, _order
        self.data = [np.array([])] * len(_imgs)
        self.plot_x, self.plot_y = _plot_x, _plot_y
        self.r_x, self.r_y = [], []
        self.clear_plots = _clear_plots
        self.plot_mode = _plot_mode
        if _pcs is None:
            _pcs = [True, False, False, False, False]
        self.pcs = _pcs
        self.correlation_mode = _correlation_mode
        self.boxed = _boxed
        self.show_legend = _show_legend
        self.refresh()

    def operations(self, x: np.ndarray) -> np.ndarray:
        # ORDER
        # 1. Устранение выбросов > масштабирование > сглаживание
        # 2. Устранение выбросов > сглаживание > масштабирование
        # 3. Сглаживание > устранение выбросов > масштабирование
        if self.order == 1:
            if self.elim:
                # print('Устранение выбросов... ', end='')
                x = co.elim_outliers(x)
                # print(' OK.')
            if self.scale:
                # print('Масштабирование... ', end='')
                x = co.scaling(x)
                # print(' OK.')
            if self.smoothing:
                # print('Сглаживание... ', end='')
                x = co.smoothing(x, self.smoothing, self.exclusive)
                # print(' OK.')
        if self.order == 2:
            if self.elim:
                # print('Устранение выбросов... ', end='')
                x = co.elim_outliers(x)
                # print(' OK.')
            if self.smoothing:
                # print('Сглаживание... ', end='')
                x = co.smoothing(x, self.smoothing, self.exclusive)
                # print(' OK.')
            if self.scale:
                # print('Масштабирование... ', end='')
                x = co.scaling(x)
                # print(' OK.')
        if self.order == 3:
            if self.smoothing:
                # print('Сглаживание... ', end='')
                x = co.smoothing(x, self.smoothing, self.exclusive)
                # print(' OK.')
            if self.elim:
                # print('Устранение выбросов... ', end='')
                x = co.elim_outliers(x)
                # print(' OK.')
            if self.scale:
                # print('Масштабирование... ', end='')
                x = co.scaling(x)
                # print(' OK.')
        return x

    def refresh(self) -> None:
        print('\n--------------------------------------------------')
        print('Применение операций. Подождите... ')
        for k, img in enumerate(imgs):
            self.data[k] = RawConverter.convert(raw=img.raw, _return_type=self.mode,
                                                operations=lambda x: self.operations(x))
            path = img.get_default_path_to_png(self.mode)
            co.save_png(self.data[k], path)
            self.imfs[k].reload_image(path)
        print(' OK.')
        print('\nИзображение #1 (целиком). Анализ')
        print('\tMin: {:.2f}'.format(np.min(self.data[0])), end='')
        print('\tAvg: {:.2f}'.format(np.mean(self.data[0])), end='')
        print('\tMax: {:.2f}'.format(np.max(self.data[0])))
        # print('\tДисперсия: {:.2f}'.format(np.var(self.data[0])), end='')
        print('\tСтд. отклонение: {:.2f}'.format(np.std(self.data[0])))
        print('Изображение #2 (целиком). Анализ')
        print('\tMin: {:.2f}'.format(np.min(self.data[1])), end='')
        print('\tAvg: {:.2f}'.format(np.mean(self.data[1])), end='')
        print('\tMax: {:.2f}'.format(np.max(self.data[1])))
        # print('\tДисперсия: {:.2f}'.format(np.var(self.data[1])), end='')
        print('\tСтд. отклонение: {:.2f}'.format(np.std(self.data[1])))

    def set(self,
            mode: DType = None, change_scale=False, change_elim=False,
            smoothing: int = None, change_exclusive=False, order: int = None,
            change_plot_x=False, change_plot_y=False, change_clear_plots=False,
            plot_mode: DType = None, _pcs: list = None,
            corr_mode: CorrelationMode = None, change_boxed=False,
            refresh=True, change_show_legend=False) -> None:
        if mode:
            self.mode = mode
        if change_scale:
            self.scale = not self.scale
        if change_elim:
            self.elim = not self.elim
        if smoothing is not None:
            self.smoothing = smoothing
        if change_exclusive:
            self.exclusive = not self.exclusive
        if order is not None:
            self.order = order
        if change_plot_x:
            self.plot_x = not self.plot_x
        if change_plot_y:
            self.plot_y = not self.plot_y
        if change_clear_plots:
            self.clear_plots = not self.clear_plots
        if plot_mode:
            self.plot_mode = plot_mode
        if _pcs:
            self.pcs = _pcs
        if corr_mode:
            self.correlation_mode = corr_mode
        if change_boxed:
            self.boxed = not self.boxed
            for imf in self.imfs:
                imf.boxed = self.boxed
        if change_show_legend:
            self.show_legend = not self.show_legend
        if refresh:
            self.refresh()

    def __construct_label(self):
        label = ''
        elim = 'нет'
        if self.elim:
            elim = 'да'
        scale = 'нет'
        if self.scale:
            scale = '0-255'
        smoothing = 'нет'
        if self.smoothing:
            smoothing = 'n={}'.format(self.smoothing)
        if self.plot_mode != DType.AS_IS:
            mode = self.mode.value
        else:
            mode = 'компл'
        if self.order == 1:
            label = 'Режим: {}. {}.\n1) Устр. выб. ({}). 2) Масштаб. ({}).\n3) Сглаживание ({}).'.format(
                mode, self.correlation_mode.value, elim, scale, smoothing
            )
        if self.order == 2:
            label = 'Режим: {}. {}.\n1) Устр. выб. ({}). 2) Сглаживание ({}).\n3) Масштаб. ({}).'.format(
                mode, self.correlation_mode.value, elim, smoothing, scale
            )
        if self.order == 3:
            label = 'Режим: {}. {}.\n1) Сглаживание ({}). 2) Устр. выб. ({}).\n3) Масштаб. ({})'.format(
                mode, self.correlation_mode.value, smoothing, elim, scale
            )
        return label

    def plot(self, shift: Union[Tuple[int, int], List[int]]):
        if self.clear_plots:
            self.r_x.clear()
            self.r_y.clear()

        coords = self.imfs[0].get_selected_area_coords()
        print('\n--------------------------------------------------')
        print('Выбрано окно [{} -> {}].'.format(*coords))
        first, second = coords
        x, y = [first[0], second[0]], [first[1], second[1]]
        print('Размер окна: {}x{}'.format(abs(x[1] - x[0]), abs(y[1] - y[0])))
        print('Сдвиг изображения II по оси X: {}'.format(shift[0]))
        print('Сдвиг изображения II по оси Y: {}'.format(shift[1]))
        print('Режим: {}'.format(self.correlation_mode.value))

        label = self.__construct_label()

        if self.plot_mode != DType.AS_IS:
            print('Данные: {}'.format(self.mode.value))
            _data = self.data
        else:
            print('Данные: комплексное изображение')
            _data = [self.operations(self.imgs[k].raw) for k in range(len(self.imgs))]

        plt.close()
        if self.plot_x:
            plt.figure(num='Корреляция по X', figsize=(3, 2), dpi=300)
            ax = plt.gca()
            ax.set_ylabel('Значения', fontweight='bold')
            ax.set_xlabel('Ось X', fontweight='bold')
            
            r = co.correlate(_data, coords,
                             direction=Direction.X, shift=shift,
                             mode=self.correlation_mode)[Direction.X.value]
            
            if self.plot_mode != DType.AS_IS:
                self.r_x.append([r, label])
            else:
                if self.pcs[0]:
                    self.r_x.append([np.array([[pos, np.absolute(val)] for pos, val in r]),
                                     '{}\nАмплитуда компл. коэфф. корреляции\n'.format(label)])
                if self.pcs[1]:
                    self.r_x.append([np.array([[pos, np.angle(val)] for pos, val in r]),
                                     '{}\nФаза компл. коэфф. корреляции (рад.)\n'.format(label)])
                if self.pcs[2]:
                    self.r_x.append([np.array([[pos, np.real(val)] for pos, val in r]),
                                     '{}\nДейств. часть компл. коэфф. корреляции\n'.format(label)])
                if self.pcs[3]:
                    self.r_x.append([np.array([[pos, np.imag(val)] for pos, val in r]),
                                     '{}\nМнимая часть компл. коэфф. корреляции\n'.format(label)])
                if self.pcs[4]:
                    self.r_x.append([np.array([[pos, np.absolute(val) * np.absolute(val)] for pos, val in r]),
                                     '{}\nИнтенсивность компл. коэфф. корреляции\n'.format(label)])

            for r, lbl in self.r_x:
                ax.plot(r[:, 0], r[:, 1], label=lbl, linewidth=0.5)
            if self.show_legend:
                ax.legend(loc='best', frameon=False, prop={'size': 4})

        if self.plot_y:
            plt.figure(num='Корреляция по Y', figsize=(3, 2), dpi=300)
            ax = plt.gca()
            ax.set_ylabel('Значения', fontweight='bold')
            ax.set_xlabel('Ось Y', fontweight='bold')

            r = co.correlate(_data, coords,
                             direction=Direction.Y, shift=shift,
                             mode=self.correlation_mode)[Direction.Y.value]

            if self.plot_mode != DType.AS_IS:
                self.r_y.append([r, label])
            else:
                if self.pcs[0]:
                    self.r_y.append([np.array([[pos, np.absolute(val)] for pos, val in r]),
                                     '{}.\nАмплитуда компл. коэфф. корреляции\n'.format(label)])
                if self.pcs[1]:
                    self.r_y.append([np.array([[pos, np.angle(val)] for pos, val in r]),
                                     '{}.\nФаза компл. коэфф. корреляции (рад.)\n'.format(label)])
                if self.pcs[2]:
                    self.r_y.append([np.array([[pos, np.real(val)] for pos, val in r]),
                                     '{}.\nДейств. часть компл. коэфф. корреляции\n'.format(label)])
                if self.pcs[3]:
                    self.r_y.append([np.array([[pos, np.imag(val)] for pos, val in r]),
                                     '{}.\nМнимая часть компл. коэфф. корреляции\n'.format(label)])
                if self.pcs[4]:
                    self.r_y.append([np.array([[pos, np.absolute(val) * np.absolute(val)] for pos, val in r]),
                                     '{}.\nИнтенсивность компл. коэфф. корреляции\n'.format(label)])

            for r, lbl in self.r_y:
                ax.plot(r[:, 0], r[:, 1], label=lbl, linewidth=0.5)
            if self.show_legend:
                ax.legend(loc='best', frameon=False, prop={'size': 4})

        plt.show()


if __name__ == "__main__":
    warnings.filterwarnings(action='ignore', category=np.ComplexWarning)

    root = Tk()
    root.title('KrossPolar [dev] - Выбор диапазона')
    root.geometry('{:.0f}x{:.0f}'.format(400, 150))
    root.resizable(width=False, height=False)
    label_f = Label(root, text='Выберите один из возможных вариантов:')
    label_f.place(relx=.5, rely=.2, anchor="center")
    band = StringVar(root, value=Band.P.name)
    r_P = Radiobutton(root, text='P-диапазон', variable=band, value=Band.P.name)
    r_P.place(relx=.2, rely=.5, anchor="center")
    r_L = Radiobutton(root, text='L-диапазон', variable=band, value=Band.L.name)
    r_L.place(relx=.7, rely=.5, anchor="center")
    ok = BooleanVar(root, value=False)
    button = Button(root, text="OK", command=lambda: ok.set(True), width=10, height=1)
    button.place(relx=.5, rely=.8, anchor="center")
    button.wait_variable(ok)
    band = Band[band.get()]
    root.destroy()

    root = Tk()
    root.geometry('{:.0f}x{:.0f}'.format(imf_width * 1.08 * 2, imf_height * 1.35))
    root.resizable(width=False, height=False)

    root.title('KrossPolar [dev] - Чтение файлов - {}-диапазон'.format(band.value))
    paths = []
    root.filename = filedialog.askopenfilename(
        initialdir=os.path.join('DATA', band.value),
        title="Выберите первое изображение",
        filetypes=[("CMP files (комплексная матрица РЛИ)", "*.cmp")])
    paths.append(root.filename)
    root.filename = filedialog.askopenfilename(
        initialdir=os.path.join('DATA', band.value),
        title="Выберите второе изображение",
        filetypes=[("CMP files (комплексная матрица РЛИ)", "*.cmp")])
    paths.append(root.filename)
    if not all(paths):
        print('Не все файлы выбраны')
        exit(1)

    # DEBUG
    # band = Band.P
    # paths = [os.path.join('DATA', 'P', 'kp3d_hv.cmp'), os.path.join('DATA', 'P', 'kp3d_vh.cmp')]
    # !DEBUG

    imgs = [RadarImage(band, path, use_dumps=True) for path in paths]

    lbls = [re.split(r'[\\/]', path[:-4])[-1] for path in paths]
    root.title('KrossPolar [dev] - Изображения {} и {}'.format(*lbls))

    imfs = [ImageField(root, imf_width, imf_height), ImageField(root, imf_width, imf_height)]
    imfs[0].connect(imfs[1])
    imfs[1].connect(imfs[0])
    imfs[0].grid(row=0, column=0)
    imfs[1].grid(row=0, column=1, columnspan=2)

    view = View(imgs, imfs)

    main_menu = Menu(root)
    root.config(menu=main_menu)
    menu = [Menu(main_menu, tearoff=0), Menu(main_menu, tearoff=0), Menu(main_menu, tearoff=0)]

    correlation_mode = StringVar(root, value=CorrelationMode.MODE1.name)
    menu[0].add_radiobutton(label='Метод 1 ( + | □ )', variable=correlation_mode,
                            value=CorrelationMode.MODE1.name,
                            command=lambda: view.set(corr_mode=CorrelationMode.MODE1, refresh=False))
    menu[0].add_radiobutton(label='Метод 2 ( □ | □ )', variable=correlation_mode,
                            value=CorrelationMode.MODE2.name,
                            command=lambda: view.set(corr_mode=CorrelationMode.MODE2, refresh=False))
    menu[0].add_separator()

    view_mode = StringVar(root, value=DType.AMPLITUDES.name)
    menu[0].add_radiobutton(label='Амплитуды', variable=view_mode, value=DType.AMPLITUDES.name,
                            command=lambda: view.set(mode=DType.AMPLITUDES))
    menu[0].add_radiobutton(label='Фазы в радианах', variable=view_mode, value=DType.PHASES.name,
                            command=lambda: view.set(mode=DType.PHASES))
    menu[0].add_radiobutton(label='Реальная часть', variable=view_mode, value=DType.REAL_PARTS.name,
                            command=lambda: view.set(mode=DType.REAL_PARTS))
    menu[0].add_radiobutton(label='Мнимая часть', variable=view_mode, value=DType.IMAG_PARTS.name,
                            command=lambda: view.set(mode=DType.IMAG_PARTS))
    menu[0].add_radiobutton(label='Интенсивности', variable=view_mode, value=DType.INTENSITIES.name,
                            command=lambda: view.set(mode=DType.INTENSITIES))
    main_menu.add_cascade(label='Режимы', menu=menu[0])

    view_elim = BooleanVar(root, value=True)
    menu[1].add_checkbutton(label='Устранение выбросов (межквартильное расстояние)', variable=view_elim,
                            command=lambda: view.set(change_elim=True))
    view_scale = BooleanVar(root, value=True)
    menu[1].add_checkbutton(label='Масштабирование 0-255', variable=view_scale,
                            command=lambda: view.set(change_scale=True))

    menu[1].add_separator()
    view_smoothing = IntVar(root, value=0)
    menu[1].add_radiobutton(label='Без сглаживания', variable=view_smoothing, value=0,
                            command=lambda: view.set(smoothing=0))

    menu[1].add_radiobutton(label='1x1', variable=view_smoothing, value=1,
                            command=lambda: view.set(smoothing=1))
    menu[1].add_radiobutton(label='2x2', variable=view_smoothing, value=2,
                            command=lambda: view.set(smoothing=2))
    menu[1].add_radiobutton(label='3x3', variable=view_smoothing, value=3,
                            command=lambda: view.set(smoothing=3))
    menu[1].add_radiobutton(label='5x5', variable=view_smoothing, value=5,
                            command=lambda: view.set(smoothing=5))
    menu[1].add_radiobutton(label='7x7', variable=view_smoothing, value=7,
                            command=lambda: view.set(smoothing=7))
    menu[1].add_radiobutton(label='10x10', variable=view_smoothing, value=10,
                            command=lambda: view.set(smoothing=10))
    menu[1].add_radiobutton(label='15x15', variable=view_smoothing, value=15,
                            command=lambda: view.set(smoothing=15))
    menu[1].add_radiobutton(label='20x20', variable=view_smoothing, value=20,
                            command=lambda: view.set(smoothing=20))
    # menu[1].add_radiobutton(label='30x30', variable=view_smoothing, value=30,
    #                         command=lambda: view.set(smoothing=30))
    # menu[1].add_radiobutton(label='50x50', variable=view_smoothing, value=50,
    #                         command=lambda: view.set(smoothing=50))
    # menu[1].add_radiobutton(label='100x100', variable=view_smoothing, value=100,
    #                         command=lambda: view.set(smoothing=100))
    menu[1].add_separator()

    view_smoothing_exclusive = BooleanVar(root, value=False)
    menu[1].add_checkbutton(label='Исключающее среднее', variable=view_smoothing_exclusive,
                            command=lambda: view.set(change_exclusive=True))

    menu[1].add_separator()
    submenu = [Menu(menu[1])]
    view_order = IntVar(root, value=1)
    submenu[0].add_radiobutton(label='Устранение выбросов > масштабирование > сглаживание',
                            variable=view_order, value=1, command=lambda: view.set(order=1))
    submenu[0].add_radiobutton(label='Устранение выбросов > сглаживание > масштабирование',
                            variable=view_order, value=2, command=lambda: view.set(order=2))
    submenu[0].add_radiobutton(label='Сглаживание > устранение выбросов > масштабирование',
                            variable=view_order, value=3, command=lambda: view.set(order=3))
    menu[1].add_cascade(label='Порядок операций', menu=submenu[0])

    plot_x_direction = BooleanVar(value=True)
    menu[2].add_checkbutton(label='Корреляция по X', variable=plot_x_direction,
                            command=lambda: view.set(change_plot_x=True, refresh=False))
    plot_y_direction = BooleanVar(value=False)
    menu[2].add_checkbutton(label='Корреляция по Y', variable=plot_y_direction,
                            command=lambda: view.set(change_plot_y=True, refresh=False))
    menu[2].add_separator()
    plot_complex = BooleanVar(value=False)
    menu[2].add_radiobutton(label='Расчет согласно выбранному режиму',
                            variable=plot_complex, value=False,
                            command=lambda: view.set(plot_mode=DType[view_mode.get()], refresh=False))
    menu[2].add_radiobutton(label='Расчет комплексного коэффициента корреляции (*)',
                            variable=plot_complex, value=True,
                            command=lambda: view.set(plot_mode=DType.AS_IS, refresh=False))
    menu[2].add_separator()
    submenu.append(Menu(menu[2]))
    pcs = []
    for i in range(5):
        pcs.append(BooleanVar(value=False))
    pcs[0].set(value=True)
    submenu[1].add_checkbutton(label='Вывести амплитуду комлп. коэфф. корреляции', variable=pcs[0],
                            command=lambda: view.set(_pcs=[_p.get() for _p in pcs], refresh=False))
    submenu[1].add_checkbutton(label='Вывести фазу (в радианах)', variable=pcs[1],
                            command=lambda: view.set(_pcs=[_p.get() for _p in pcs], refresh=False))
    submenu[1].add_checkbutton(label='Вывести реальную часть', variable=pcs[2],
                            command=lambda: view.set(_pcs=[_p.get() for _p in pcs], refresh=False))
    submenu[1].add_checkbutton(label='Вывести мнимую часть', variable=pcs[3],
                            command=lambda: view.set(_pcs=[_p.get() for _p in pcs], refresh=False))
    submenu[1].add_checkbutton(label='Вывести интенсивность', variable=pcs[4],
                            command=lambda: view.set(_pcs=[_p.get() for _p in pcs], refresh=False))
    menu[2].add_cascade(label='Дополнительные настройки для (*)', menu=submenu[1])
    
    menu[2].add_separator()
    plot_clear = BooleanVar(value=False)
    menu[2].add_checkbutton(label='Очистить предыдущие результаты', variable=plot_clear,
                            command=lambda: view.set(change_clear_plots=True, refresh=False))

    menu[2].add_separator()
    plot_show_legend = BooleanVar(value=True)
    menu[2].add_checkbutton(label='Показывать легенду', variable=plot_show_legend,
                            command=lambda: view.set(change_show_legend=True, refresh=False))
    main_menu.add_cascade(label='Графики', menu=menu[2])

    main_menu.add_cascade(label='Настройки', menu=menu[1])

    x_shift = IntVar(value=1)
    x_shift_scale = Scale(root, orient=HORIZONTAL, from_=-3, to=3, tickinterval=3, resolution=1,
                          variable=x_shift)
    x_shift_scale.grid(row=1, column=1, rowspan=2)
    y_shift_scale = Scale(root, orient=VERTICAL, from_=-3, to=3, tickinterval=3, resolution=1)
    y_shift_scale.grid(row=1, column=2, rowspan=2)

    boxed = BooleanVar(value=False)
    c_box = Checkbutton(root, text='Квадратная область', variable=boxed,
                        command=lambda: view.set(change_boxed=True, refresh=False))
    c_box.grid(row=1, column=0)

    b_compute = Button(root, text="Вычислить", width=20, height=1)
    b_compute.config(command=lambda: view.plot([x_shift_scale.get(), y_shift_scale.get()]))
    b_compute.grid(row=2, column=0)

    root.mainloop()
