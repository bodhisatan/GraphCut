#include "graphcutimpl.h"

#define BIGINT 10000000
#define LARGE_NUMBER 100000
const double gamma = 50;
double beta;
bool if_Lclicked = false;
bool if_Rclicked = false;
const Vec3b blac(0, 0, 0);
Mat image, gray_image, selected_record, copy_of_image;
const string WinName = "Graph-Cut�㷨";
Rect WinRect;
int image_cols, image_rows;
//Ŀ���Լ�������ֱ��ͼ
int sum_of_object_pixels = 0;
int sum_of_background_pixels = 0;
double object_histogram[256] = { 0 };
double background_histogram[256] = { 0 };
double value_of_right[BIGINT] = { 0 }, value_of_down[BIGINT] = { 0 }, value_of_rd[BIGINT] = { 0 }, value_of_ru[BIGINT] = { 0 };

void add2histogram(int col, int row, int& sum, double histogram[256], int OorB)
{
	for (int i = -4; i <= 4; i++)
		for (int j = -4; j <= 4; j++)
		{
			if (WinRect.contains(Point(col + i, row + j)))  //����ʾ��Ļ��
			{
				if (selected_record.at<uchar>(row + i, col + j) == 255)  //δ����ǹ�
				{
					selected_record.at<uchar>(row + i, col + j) = OorB;     //�����
					sum++;                                              //ֱ��ͼ��ǵ�����������һ
					histogram[gray_image.at<uchar>(row + i, col + j)]++;    //�������ڻҶȼ�����һ
				}
			}
		}

}



static void onmouse(int event, int x, int y, int flag, void* param)
{
	switch (event)
	{
	case EVENT_LBUTTONDOWN:
		if_Lclicked = true;
		break;

	case EVENT_RBUTTONDOWN:
		if_Rclicked = true;
		break;


	case EVENT_LBUTTONUP:
		if_Lclicked = false;
		break;

	case EVENT_RBUTTONUP:
		if_Rclicked = false;
		break;

	case EVENT_MOUSEMOVE:
		//add2histogram(x, y);
		if (if_Lclicked)
		{
			circle(image, Point(x, y), 4, CV_RGB(255, 0, 0), 4);
			imshow(WinName, image);
			add2histogram(x, y, sum_of_object_pixels, object_histogram, 0);
			//cout << sum_of_object_pixels << endl;
			//for (int i = 0; i <= 255; i++)
			//    cout << i << ' ' << object_histogram[i] << endl;
		}
		if (if_Rclicked)
		{
			circle(image, Point(x, y), 4, CV_RGB(0, 255, 0), 4);
			imshow(WinName, image);
			add2histogram(x, y, sum_of_background_pixels, background_histogram, 1);
			//cout << sum_of_object_pixels << endl;
			//for (int i = 0; i <= 255; i++) if (object_histogram[i] != 0) cout << object_histogram[i];
		}
		break;
	}
}

double Distance(Vec3d a, Vec3d b)
{
	return (a[0] - b[0]) * (a[0] - b[0]) + (a[1] - b[1]) * (a[1] - b[1]) + (a[2] - b[2]) * (a[2] - b[2]);

}

void graph_cut_work()
{
	double link2sourse, link2sink;
	//double value_of_down, value_of_right;
	typedef Graph<double, double, double> GraphType;
	GraphType* g = new GraphType(image_rows * image_cols, image_rows * image_cols * 5);
	//����Ϊ����ͼ��������graph.h
	g->add_node(image_rows * image_cols);

	//ֱ��ͼƽ����
	double object_histogram_tem[256], background_histogram_tem[256];
	for (int i = 0; i < 256; i++)
	{
		object_histogram_tem[i] = object_histogram[i];
		background_histogram_tem[i] = background_histogram[i];
	}
	for (int i = 1; i < 255; i++)
	{
		object_histogram[i] = 0.66 * object_histogram[i] + 0.17 * object_histogram[i - 1] + 0.17 * object_histogram[i + 1];
		background_histogram[i] = 0.66 * background_histogram[i] + 0.17 * background_histogram[i - 1] + 0.17 * background_histogram[i + 1];
	}

	//����һ������Ҷ�������������
	for (int i = 0; i < image_rows; i++)
		for (int j = 0; j < image_cols; j++)
		{
			if (selected_record.at<uchar>(i, j) == 0)
			{
				g->add_tweights(i * image_cols + j, LARGE_NUMBER, 0);
				continue;
			}
			if (selected_record.at<uchar>(i, j) == 1)
			{
				g->add_tweights(i * image_cols + j, 0, LARGE_NUMBER);
				continue;
			}
			//���������,��δ��ע������
			link2sink = -log(((double)(0.1 + object_histogram[gray_image.at<uchar>(i, j)]) / (double)(sum_of_object_pixels + 1)));
			link2sourse = -log(((double)(0.1 + background_histogram[gray_image.at<uchar>(i, j)]) / (double)(sum_of_background_pixels + 1)));
			//cout << i << ' ' << j << ' ' << link2sourse << ' ' << link2sink << endl;
			//cout << link2sink << ' ' << link2sourse << endl;
			g->add_tweights(i * image_cols + j, link2sourse, link2sink);
		}

	//����һ�������Ե������������
	for (int i = 1; i < image_rows - 1; i++)
		for (int j = 1; j < image_cols - 1; j++)
		{
			value_of_right[i * image_cols + j] = gamma * exp(-beta * Distance(image.at<Vec3b>(i, j), image.at<Vec3b>(i, j + 1)));
			value_of_down[i * image_cols + j] = gamma * exp(-beta * Distance((Vec3d)image.at<Vec3b>(i, j), (Vec3d)image.at<Vec3b>(i + 1, j)));
			value_of_rd[i * image_cols + j] = gamma / 1.414 * exp(-beta * Distance((Vec3d)image.at<Vec3b>(i, j), (Vec3d)image.at<Vec3b>(i + 1, j + 1)));
			value_of_ru[i * image_cols + j] = gamma / 1.414 * exp(-beta * Distance((Vec3d)image.at<Vec3b>(i, j), (Vec3d)image.at<Vec3b>(i - 1, j + 1)));
		}
	for (int i = 1; i < image_rows - 1; i++)
		for (int j = 1; j < image_cols - 1; j++)
		{
			g->add_edge(i * image_cols + j, (i + 1) * image_cols + j, value_of_down[i * image_cols + j], value_of_down[i * image_cols + j]);
			g->add_edge(i * image_cols + j, i * image_cols + j + 1, value_of_right[i * image_cols + j], value_of_right[i * image_cols + j]);
			g->add_edge(i * image_cols + j, (i + 1) * image_cols + j + 1, value_of_rd[i * image_cols + j], value_of_rd[i * image_cols + j]);
			g->add_edge(i * image_cols + j, (i - 1) * image_cols + j + 1, value_of_ru[i * image_cols + j], value_of_ru[i * image_cols + j]);
		}
	//���ڱ�Ե����ʡ�ԣ����Խ��
	copy_of_image.copyTo(image);
	int flow = g->maxflow();
	for (int i = 0; i < image_rows; i++)
		for (int j = 0; j < image_cols; j++)
		{
			if (g->what_segment(i * image_cols + j) != GraphType::SOURCE)
			{
				gray_image.at<uchar>(i, j) = 255;
				image.at<Vec3b>(i, j) = blac;
			}
		}

	imshow(WinName, image);
}



double calculate_Beta(Mat& image)
{
	for (int y = 1; y < image.rows - 1; y++)
	{
		for (int x = 1; x < image.cols - 1; x++)
		{
			Vec3d color = image.at<Vec3b>(y, x);
			Vec3d delta = color - (Vec3d)image.at<Vec3b>(y, x - 1);
			beta += delta.dot(delta);
			delta = color - (Vec3d)image.at<Vec3b>(y - 1, x);
			beta += delta.dot(delta);
			delta = color - (Vec3d)image.at<Vec3b>(y - 1, x - 1);
			beta += delta.dot(delta);
			delta = color - (Vec3d)image.at<Vec3b>(y - 1, x + 1);
			beta += delta.dot(delta);
		}
	}

	return (2 * image.cols * image.rows - 4 * image.cols - 4 * image.rows + 9) / beta;
}

void graph_cut_init(std::string path)
{
	image = imread(path, IMREAD_COLOR);
	beta = calculate_Beta(image);  //��beta

	image.copyTo(copy_of_image); //image�Ŀ���
	image_cols = image.cols;  //��������
	image_rows = image.rows;  //������߶�

	WinRect = Rect(0, 0, image.cols, image_rows);
	cvtColor(image, gray_image, CV_BGR2GRAY);
	selected_record = Mat(image_rows, image_cols, CV_8UC1, 255); //�ȸ߶��ٿ��
	//namedWindow(WinName, WINDOW_AUTOSIZE);
	imshow(WinName, image);

	setMouseCallback(WinName, onmouse, 0);

}