#include "grabcutimpl.h"

clock_t start, endt;
Mat image_copy, imager, Maskr;
#define BGC 0
#define FGC 255
#define MBGC 111
#define MFGC 150
#define LARGE_NUMBER 10000
#define BIGINT 10000000
double _value_of_right[BIGINT] = { 0 }, _value_of_down[BIGINT] = { 0 }, _value_of_rd[BIGINT] = { 0 }, _value_of_ru[BIGINT] = { 0 };
const Vec3b blac(0, 0, 0);
const double _gamma = 50;
double _beta = 0;
double link2Source, link2Sink;
int Bigflag = 0;
//
const string WinName = "Grab-Cut算法"; //窗口名设置
Rect rect_tem;  //本来是临时矩阵,现在用来优化速度
Mat Mask;  //用户输入部分关于前景，后景的标记
Mat _image;
int _image_cols, _image_rows; //图像的宽高
//
typedef Graph<double, double, double> GraphType;


GMM::GMM(int i)
{

	const int modelSize = 3 + 9 + 1;
	model.create(1, modelSize * componentsCount, CV_64FC1);
	model.setTo(Scalar(0));
	coefs = model.ptr<double>(0);
	mean = coefs + componentsCount;
	cov = mean + 3 * componentsCount;
	for (int ci = 0; ci < componentsCount; ci++)
		if (coefs[ci] > 0)
			calcInverseCovAndDeterm(ci);
}
double GMM::operator()(const Vec3d color) const
{
	double res = 0;
	for (int ci = 0; ci < componentsCount; ci++)
		res += coefs[ci] * (*this)(ci, color);
	return res;
}
double GMM::operator()(int ci, const Vec3d color) const
{
	double res = 0;
	if (coefs[ci] > 0)
	{
		Vec3d diff = color;
		double* m = mean + 3 * ci;
		diff[0] -= m[0]; diff[1] -= m[1]; diff[2] -= m[2];
		double mult = diff[0] * (diff[0] * inverseCovs[ci][0][0] + diff[1] * inverseCovs[ci][1][0] + diff[2] * inverseCovs[ci][2][0])
			+ diff[1] * (diff[0] * inverseCovs[ci][0][1] + diff[1] * inverseCovs[ci][1][1] + diff[2] * inverseCovs[ci][2][1])
			+ diff[2] * (diff[0] * inverseCovs[ci][0][2] + diff[1] * inverseCovs[ci][1][2] + diff[2] * inverseCovs[ci][2][2]);
		res = 1.0f / sqrt(covDeterms[ci]) * exp(-0.5f * mult);
	}
	return res;
}
int GMM::whichComponent(const Vec3d color) const
{
	int k = 0;
	double max = 0;

	for (int ci = 0; ci < componentsCount; ci++)
	{
		double p = (*this)(ci, color);
		if (p > max)
		{
			k = ci;
			max = p;
		}
	}
	return k;
}
void GMM::initLearning()
{
	for (int ci = 0; ci < componentsCount; ci++)
	{
		sums[ci][0] = sums[ci][1] = sums[ci][2] = 0;
		prods[ci][0][0] = prods[ci][0][1] = prods[ci][0][2] = 0;
		prods[ci][1][0] = prods[ci][1][1] = prods[ci][1][2] = 0;
		prods[ci][2][0] = prods[ci][2][1] = prods[ci][2][2] = 0;
		sampleCounts[ci] = 0;
	}
	totalSampleCount = 0;
}
void GMM::addSample(int ci, const Vec3d color)
{
	sums[ci][0] += color[0]; sums[ci][1] += color[1]; sums[ci][2] += color[2];
	prods[ci][0][0] += color[0] * color[0]; prods[ci][0][1] += color[0] * color[1]; prods[ci][0][2] += color[0] * color[2];
	prods[ci][1][0] += color[1] * color[0]; prods[ci][1][1] += color[1] * color[1]; prods[ci][1][2] += color[1] * color[2];
	prods[ci][2][0] += color[2] * color[0]; prods[ci][2][1] += color[2] * color[1]; prods[ci][2][2] += color[2] * color[2];
	sampleCounts[ci]++;
	totalSampleCount++;
}
void GMM::endLearning()
{
	const double variance = 0.01;
	for (int ci = 0; ci < componentsCount; ci++)
	{
		int n = sampleCounts[ci];
		if (n == 0)
			coefs[ci] = 0;
		else
		{

			coefs[ci] = (double)n / totalSampleCount;
			double* m = mean + 3 * ci;
			m[0] = sums[ci][0] / n; m[1] = sums[ci][1] / n; m[2] = sums[ci][2] / n;
			double* c = cov + 9 * ci;


			//
			//
			//
			cout << ci << ":" << coefs[ci] << '*' << m[0] << ' ' << m[1] << ' ' << m[2] << endl;
			//
			//
			//





			c[0] = prods[ci][0][0] / n - m[0] * m[0]; c[1] = prods[ci][0][1] / n - m[0] * m[1]; c[2] = prods[ci][0][2] / n - m[0] * m[2];
			c[3] = prods[ci][1][0] / n - m[1] * m[0]; c[4] = prods[ci][1][1] / n - m[1] * m[1]; c[5] = prods[ci][1][2] / n - m[1] * m[2];
			c[6] = prods[ci][2][0] / n - m[2] * m[0]; c[7] = prods[ci][2][1] / n - m[2] * m[1]; c[8] = prods[ci][2][2] / n - m[2] * m[2];
			double dtrm = c[0] * (c[4] * c[8] - c[5] * c[7]) - c[1] * (c[3] * c[8] - c[5] * c[6]) + c[2] * (c[3] * c[7] - c[4] * c[6]);
			if (dtrm <= std::numeric_limits<double>::epsilon())
			{
				c[0] += variance;
				c[4] += variance;
				c[8] += variance;
			}
			calcInverseCovAndDeterm(ci);
		}
	}
}
void GMM::calcInverseCovAndDeterm(int ci)
{
	if (coefs[ci] > 0)
	{
		double* c = cov + 9 * ci;
		double dtrm =
			covDeterms[ci] = c[0] * (c[4] * c[8] - c[5] * c[7]) - c[1] * (c[3] * c[8] - c[5] * c[6])
			+ c[2] * (c[3] * c[7] - c[4] * c[6]);
		//三阶方阵的求逆
		inverseCovs[ci][0][0] = (c[4] * c[8] - c[5] * c[7]) / dtrm;
		inverseCovs[ci][1][0] = -(c[3] * c[8] - c[5] * c[6]) / dtrm;
		inverseCovs[ci][2][0] = (c[3] * c[7] - c[4] * c[6]) / dtrm;
		inverseCovs[ci][0][1] = -(c[1] * c[8] - c[2] * c[7]) / dtrm;
		inverseCovs[ci][1][1] = (c[0] * c[8] - c[2] * c[6]) / dtrm;
		inverseCovs[ci][2][1] = -(c[0] * c[7] - c[1] * c[6]) / dtrm;
		inverseCovs[ci][0][2] = (c[1] * c[5] - c[2] * c[4]) / dtrm;
		inverseCovs[ci][1][2] = -(c[0] * c[5] - c[2] * c[3]) / dtrm;
		inverseCovs[ci][2][2] = (c[0] * c[4] - c[1] * c[3]) / dtrm;
	}
}




//这里是onmouse函数用到的一些标记
Point _first_point, _second_point;
bool _if_Lclicked = false;
bool _if_Rclicked = false;
bool _rect_already_selected = false;
//


//鼠标操作
//先框定一个边框，框外标记为背景，框内标记为可能的前景
//之后再选择是否进一步标注，如果进一步标注
//则左键为前景，右键为背景
static void onmouse(int event, int x, int y, int flag, void* param)
{
	if (!_rect_already_selected)
	{
		switch (event)
		{
		case EVENT_LBUTTONDOWN:
			_first_point = Point(x, y);
			//cout << x << ' ' << y << endl;
			//circle(image, Point(x, y), 4, CV_RGB(255, 0, 0), 4);
			//imshow(WinName, image);
			break;

		case EVENT_LBUTTONUP:
			//cout << x << ' ' << y << endl;
			_rect_already_selected = true;

			_second_point = Point(x, y);
			rect_tem.x = min(_first_point.x, _second_point.x);
			rect_tem.y = min(_first_point.y, _second_point.y);
			rect_tem.width = abs(_first_point.x - _second_point.x);
			rect_tem.height = abs(_first_point.y - _second_point.y);
			//cout << rect_tem << endl;
			(Mask(rect_tem)).setTo(Scalar(150)); //此处150表示可能的前景，即矩形内皆为可能的区域
			rectangle(_image, rect_tem, Scalar(255, 0, 0), 1, LINE_8);
			imshow(WinName, _image);
			break;

		}
		return;
	}
	//以下为二次标注，即在选定矩形后进一步标注其他区域的准确信息
	switch (event)
	{
	case EVENT_LBUTTONDOWN:
		_if_Lclicked = true;
		break;

	case EVENT_RBUTTONDOWN:
		_if_Rclicked = true;
		break;

	case EVENT_LBUTTONUP:
		_if_Lclicked = false;
		break;

	case EVENT_RBUTTONUP:
		_if_Rclicked = false;
		break;

	case EVENT_MOUSEMOVE:
		if (_if_Lclicked)
		{
			circle(_image, Point(x, y), 4, CV_RGB(255, 0, 0), 4);
			circle(Mask, Point(x, y), 4, Scalar(255), 4);  //确定的前景
			imshow(WinName, _image);
		}
		if (_if_Rclicked)
		{
			circle(_image, Point(x, y), 4, CV_RGB(0, 255, 0), 4);
			circle(Mask, Point(x, y), 4, Scalar(0), 4);  //确定的背景
			imshow(WinName, _image);
		}
		break;
	}
}

//输入环节
void mask_input()
{
	if (Bigflag == 0)
		Mask = Mat(_image_rows, _image_cols, CV_8UC1, Scalar(0));  //初始值全部设置为背景(0)，后面选择方框会将其覆盖
	else Mask = Maskr;
	setMouseCallback(WinName, onmouse, 0);
}

//对GMM模型利用opencv自带的kmeans进行初始化
void GMMinit(Mat& image, Mat& Mask, GMM& bgGMM, GMM& fgGMM)
{
	vector<Vec3f> bgpix, fgpix; //此处转为实数型可能是为了后面的计算
	for (int y = 0; y < _image_rows; y++)
		for (int x = 0; x < _image_cols; x++)
		{
			if (Mask.at<uchar>(Point(x, y)) == MBGC || Mask.at<uchar>(Point(x, y)) == BGC)
			{
				bgpix.push_back((Vec3f)image.at<Vec3b>(Point(x, y)));  //Vec3b是用来访问uchar向量的
			}
			else
			{
				fgpix.push_back((Vec3f)image.at<Vec3b>(Point(x, y)));
			}
		}
	Mat bg_label, fg_label; //记录每个像素对应GMM的哪个分量
	Mat bgpix_tem((int)bgpix.size(), 3, CV_32FC1, &bgpix[0][0]);   //这步是为了适应kmeans函数的格式,后面就没啥用了
	Mat fgpix_tem((int)fgpix.size(), 3, CV_32FC1, &fgpix[0][0]);

	//cout << 1;

	kmeans(bgpix_tem, 5, bg_label, TermCriteria(1, 8, 0.0), 0, 2);
	kmeans(fgpix_tem, 5, fg_label, TermCriteria(1, 8, 0.0), 0, 2);
	//上面把像素都分类好了
	//下面开始算参数
	//cout << "bgdGMM:";
	bgGMM.initLearning();
	fgGMM.initLearning();
	for (int i = 0; i < (int)bgpix.size(); i++) bgGMM.addSample(bg_label.at<int>(i, 0), bgpix[i]);
	for (int i = 0; i < (int)fgpix.size(); i++) fgGMM.addSample(fg_label.at<int>(i, 0), fgpix[i]);
	cout << "INITbgGMM:";
	bgGMM.endLearning();
	cout << "INITfgGMM:";
	fgGMM.endLearning();
}

//为像素分配自己最可能的
void reassign(Mat& image, Mat& Mask, GMM& bgGMM, GMM& fgGMM, Mat& pixel_belong_to)
{
	Point p;
	for (p.y = 0; p.y < image.rows; p.y++)
	{
		for (p.x = 0; p.x < image.cols; p.x++)
		{
			Vec3d color = image.at<Vec3b>(p);
			if (Mask.at<uchar>(p) == BGC || Mask.at<uchar>(p) == MBGC)
				pixel_belong_to.at<int>(p) = bgGMM.whichComponent(color);
			else
				pixel_belong_to.at<int>(p) = fgGMM.whichComponent(color);

		}
	}
}

//
void learnGMMs(Mat& image, Mat& Mask, Mat& belongto, GMM& bgGMM, GMM& fgGMM)
{
	bgGMM.initLearning();
	fgGMM.initLearning();
	Point p;
	for (int ci = 0; ci < 5; ci++)
	{
		for (p.y = 0; p.y < image.rows; p.y++)
		{
			for (p.x = 0; p.x < image.cols; p.x++)
			{
				if (belongto.at<int>(p) == ci)
				{
					if (Mask.at<uchar>(p) == BGC || Mask.at<uchar>(p) == MBGC)
						bgGMM.addSample(ci, image.at<Vec3b>(p));
					else
						fgGMM.addSample(ci, image.at<Vec3b>(p));
				}
			}
		}
	}
	cout << "bgGMM:";
	bgGMM.endLearning();

	cout << "fgGMM:";
	fgGMM.endLearning();
}
double _Distance(Vec3d a, Vec3d b)
{
	return (a[0] - b[0]) * (a[0] - b[0]) + (a[1] - b[1]) * (a[1] - b[1]) + (a[2] - b[2]) * (a[2] - b[2]);

}
double _calculate_Beta(Mat& image)
{
	for (int y = 1; y < image.rows - 1; y++)
	{
		for (int x = 1; x < image.cols - 1; x++)
		{
			Vec3d color = image.at<Vec3b>(y, x);
			Vec3d delta = color - (Vec3d)image.at<Vec3b>(y, x - 1);
			_beta += delta.dot(delta);
			delta = color - (Vec3d)image.at<Vec3b>(y - 1, x);
			_beta += delta.dot(delta);
			delta = color - (Vec3d)image.at<Vec3b>(y - 1, x - 1);
			_beta += delta.dot(delta);
			delta = color - (Vec3d)image.at<Vec3b>(y - 1, x + 1);
			_beta += delta.dot(delta);
		}
	}

	return (2 * image.cols * image.rows - 4 * image.cols - 4 * image.rows + 9) / _beta;
}


void Mygrabcut(Mat& image, Mat& Mask, int itcount = 3)
{

	Mat pixel_belong_to(image.size(), CV_32SC1);   //描述每个像素属于的GMM分量
	GMM bgGMM(0), fgGMM(0);      //对前景后景的GMM格式初始化
	GMMinit(image, Mask, bgGMM, fgGMM); //将image中点分别聚类
	while (itcount > 0)  //迭代次数为3次
	{
		itcount--;
		//start = clock();
		GraphType* g = new GraphType(_image_rows * _image_cols, _image_rows * _image_cols * 10);
		g->add_node(_image_rows * _image_cols);   //加入顶点个数
		//endt = clock();
		//double endtime = (double)(endt - start) / CLOCKS_PER_SEC;
		//cout << "Total time:" << endtime << endl;
		reassign(image, Mask, bgGMM, fgGMM, pixel_belong_to); //分配每个像素点
		learnGMMs(image, Mask, pixel_belong_to, bgGMM, fgGMM);  //重新产生模型参数
		//下面建图；
		//第一步，建立到特殊点的连接
		Point p;
		for (p.y = 0; p.y < _image_rows; p.y++)
		{
			for (p.x = 0; p.x < _image_cols; p.x++)
			{

				Vec3b color = image.at<Vec3b>(p);

				if (Mask.at<uchar>(p) == MBGC || Mask.at<uchar>(p) == MFGC)  //可能的前后景
				{
					link2Source = -log(bgGMM(color));
					link2Sink = -log(fgGMM(color));
				}
				else if (Mask.at<uchar>(p) == BGC)    //确定的背景
				{
					link2Source = 0;
					link2Sink = LARGE_NUMBER;
				}
				else                                 //确定的前景
				{
					link2Source = LARGE_NUMBER;
					link2Sink = 0;
				}
				g->add_tweights(p.y * _image_cols + p.x, link2Source, link2Sink);
			}
		}
		//建立普通点之间的连接
		//for (int i = 1; i < _image_rows - 1; i++)
		//for (int j = 1; j < _image_cols - 1; j++)
		for (int i = max(rect_tem.y - 3, 1); i < min(_image_rows - 1, rect_tem.y + rect_tem.height + 3); i++)
			for (int j = max(rect_tem.x - 3, 1); j < min(_image_cols - 1, rect_tem.x + rect_tem.width + 3); j++)
			{
				g->add_edge(i * _image_cols + j, (i + 1) * _image_cols + j, _value_of_down[i * _image_cols + j], _value_of_down[i * _image_cols + j]);
				g->add_edge(i * _image_cols + j, i * _image_cols + j + 1, _value_of_right[i * _image_cols + j], _value_of_right[i * _image_cols + j]);
				g->add_edge(i * _image_cols + j, (i + 1) * _image_cols + j + 1, _value_of_rd[i * _image_cols + j], _value_of_rd[i * _image_cols + j]);
				g->add_edge(i * _image_cols + j, (i - 1) * _image_cols + j + 1, _value_of_ru[i * _image_cols + j], _value_of_ru[i * _image_cols + j]);
			}


		//对g求解
		int flow = g->maxflow();
		//对mask重新染色
		//for (int i = 0; i < _image_rows; i++)
		//for (int j = 0; j < _image_cols; j++)
		for (int i = max(rect_tem.y - 3, 1); i < min(_image_rows - 1, rect_tem.y + rect_tem.height + 3); i++)
			for (int j = max(rect_tem.x - 3, 1); j < min(_image_cols - 1, rect_tem.x + rect_tem.width + 3); j++)

			{
				if (g->what_segment(i * _image_cols + j) != GraphType::SOURCE)
				{
					if (Mask.at<uchar>(i, j) != BGC) Mask.at<uchar>(i, j) = MBGC;
				}
				else
				{
					if (Mask.at<uchar>(i, j) != FGC) Mask.at<uchar>(i, j) = MFGC;
				}
			}
		cout << endl << endl << endl;
	}
}
//记忆化，这样可以加快速度，同时，为了简略起见，舍去边缘的行列
void nlink_generator()
{
	for (int i = 1; i < _image_rows - 1; i++)
		for (int j = 1; j < _image_cols - 1; j++)
		{
			_value_of_right[i * _image_cols + j] = _gamma * exp(-_beta * _Distance(_image.at<Vec3b>(i, j), _image.at<Vec3b>(i, j + 1)));
			_value_of_down[i * _image_cols + j] = _gamma * exp(-_beta * _Distance((Vec3d)_image.at<Vec3b>(i, j), (Vec3d)_image.at<Vec3b>(i + 1, j)));
			_value_of_rd[i * _image_cols + j] = _gamma / 1.414 * exp(-_beta * _Distance((Vec3d)_image.at<Vec3b>(i, j), (Vec3d)_image.at<Vec3b>(i + 1, j + 1)));
			_value_of_ru[i * _image_cols + j] = _gamma / 1.414 * exp(-_beta * _Distance((Vec3d)_image.at<Vec3b>(i, j), (Vec3d)_image.at<Vec3b>(i - 1, j + 1)));
		}
}

void GMM::grabcutInit(std::string graph_path)
{
	_image = imread(graph_path, IMREAD_COLOR);

	_image_cols = _image.cols;  //列数或宽度
	_image_rows = _image.rows;  //行数或高度
	_beta = _calculate_Beta(_image);  //算_beta
	//cout << _beta << endl;
	nlink_generator(); //记忆化,用于加速
	_image.copyTo(image_copy); //图像的复制
	_image.copyTo(imager);

	imager.copyTo(_image);
	imshow(WinName, _image);
	mask_input(); //Mask的输入
	
}

void GMM::grabcutCalc()
{
	Bigflag = 1;
	
	_image.copyTo(imager);
	Mask.copyTo(Maskr);
	image_copy.copyTo(_image); //恢复image

	Mygrabcut(_image, Mask);
	//以下是输出结果
	for (int i = 0; i < _image_rows; i++)
		for (int j = 0; j < _image_cols; j++)
		{
			if (Mask.at<uchar>(i, j) == BGC || Mask.at<uchar>(i, j) == MBGC)
				_image.at<Vec3b>(i, j) = blac;
		}
	imshow(WinName, _image);
}

