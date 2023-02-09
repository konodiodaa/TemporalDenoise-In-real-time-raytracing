#include "denoiser.h"

Denoiser::Denoiser() : m_useTemportal(false) {}

void Denoiser::Reprojection(const FrameInfo &frameInfo) {
    int height = m_accColor.m_height;
    int width = m_accColor.m_width;
    Matrix4x4 preWorldToScreen =
        m_preFrameInfo.m_matrix[m_preFrameInfo.m_matrix.size() - 1];
    Matrix4x4 preWorldToCamera =
        m_preFrameInfo.m_matrix[m_preFrameInfo.m_matrix.size() - 2];
#pragma omp parallel for
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            // TODO: Reproject
            m_valid(x, y) = false;
            m_misc(x, y) = Float3(0.f);

            int cur_id = frameInfo.m_id(x, y);
            if (cur_id == -1) {
                continue;
            }

            Matrix4x4 modelInverse = Inverse(frameInfo.m_matrix[cur_id]);
            Float3 objPos = modelInverse(frameInfo.m_position(x,y), Float3::Point); // obj in model space
            Float3 pre_ScreenPos = preWorldToScreen
                                            (m_preFrameInfo.m_matrix[cur_id](objPos,Float3::Point)
                                            , Float3::Point); // obj in screen space pre frame.

            if (pre_ScreenPos.x >=0 && pre_ScreenPos.x <= width 
                && pre_ScreenPos.y >= 0 && pre_ScreenPos.y <= height) 
            {
                int pre_id = m_preFrameInfo.m_id(pre_ScreenPos.x, pre_ScreenPos.y);
                if (pre_id == cur_id) {
                    m_misc(x, y) = m_accColor(pre_ScreenPos.x, pre_ScreenPos.y);
                    m_valid(x, y) = true;
                }
            }
        }
    }
    std::swap(m_misc, m_accColor);
}

void Denoiser::TemporalAccumulation(const Buffer2D<Float3> &curFilteredColor) {
    int height = m_accColor.m_height;
    int width = m_accColor.m_width;
    int kernelRadius = 8;
#pragma omp parallel for
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            // TODO: Temporal clamp
            Float3 color = m_accColor(x, y);
            // TODO: Exponential moving average

            if (m_valid(x, y)) 
            {
                float alpha = 1.0f;


                int left = std::max(0, x - kernelRadius);
                int right = std::min(width - 1, x + kernelRadius);
                int bottom = std::max(0, y - kernelRadius);
                int top = std::min(height - 1, y + kernelRadius);

                int num = (right - left + 1) * (top - bottom + 1);

                Float3 miu(0.0f);
                Float3 sigma(0.0f);

                for (int i = left; i <= right; i++) {
                    for (int j = bottom; j <= top; j++) {
                        miu += m_accColor(i, j);
                        sigma += Sqr(curFilteredColor(x, y) - curFilteredColor(i, j));
                    }
                }
                miu /= (float)num;
                sigma = SafeSqrt(sigma / (float)num);

                Float3 color = Clamp(color, miu - sigma * m_colorBoxK,
                                     miu + sigma * m_colorBoxK);

                m_misc(x, y) = Lerp(color, curFilteredColor(x, y), alpha);
            } 
            else
                m_misc(x, y) = curFilteredColor(x, y);
        }
    }
    std::swap(m_misc, m_accColor);
}

Buffer2D<Float3> Denoiser::Filter(const FrameInfo &frameInfo) {
    int height = frameInfo.m_beauty.m_height;
    int width = frameInfo.m_beauty.m_width;
    Buffer2D<Float3> filteredImage = CreateBuffer2D<Float3>(width, height);
    int kernelRadius = 16;
#pragma omp parallel for
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            // TODO: Joint bilateral filter
            //filteredImage(x, y) = frameInfo.m_beauty(x, y);

            int left = std::max(0, x - kernelRadius);
            int right = std::min(width-1, x + kernelRadius);
            int bottom = std::max(0, y - kernelRadius);
            int top = std::min(height - 1, y + kernelRadius);

            float weight_sum = 0.0f;

            for (int i = left;i<=right;i++) {
                for (int j = bottom; j <= top; j++) {

                    float Coord_dist = - SqrDistance(frameInfo.m_position(x, y),
                                                   frameInfo.m_position(i, j)) /
                                                   (2.0f * Sqr(m_sigmaCoord));
                    float Color_dist = - SqrDistance(frameInfo.m_beauty(x, y), 
                                                    frameInfo.m_beauty(i, j)) /
                                                    (2.0f * Sqr(m_sigmaColor));

                    // 对于一个立方体，任意相邻的两个面我们都不希望它们互相有贡献，
                    // 因此我们定义 Dnormal 项为两个法线间 的夹角 (弧度)

                    float Normal_dist = SafeAcos(Dot(frameInfo.m_normal(x, y), frameInfo.m_normal(i, j)));
                    Normal_dist = - Sqr(Normal_dist) / (2.0f * Sqr(m_sigmaNormal));

                    // 有一本书放在一张桌子上，桌 子和书的平面完全平行。
                    // 我们一定不希望书和桌子上的像素会互相贡献。
                    // 因此，我们定义这项为点 i 到 j 的单位向量与 i 点法线的点积

                    float Plane_dist = 0.0f;
                    float d = Distance(frameInfo.m_position(i, j), frameInfo.m_position(x, y));
                    if (d > 0) {
                        Plane_dist = std::max(0.0f,Dot(frameInfo.m_normal(x, y),
                               (frameInfo.m_position(i, j) - frameInfo.m_position(x, y)) / d));
                        Plane_dist *= Plane_dist;
                        Plane_dist /= -2.0f * m_sigmaPlane * m_sigmaPlane;
                    }

                   float weight = std::exp(Coord_dist + Color_dist + Normal_dist + Plane_dist);
                   weight_sum += weight;
                   filteredImage(x, y) += frameInfo.m_beauty(i, j) * weight;

                }
            }
            if (weight_sum == 0)
                filteredImage(x, y) = frameInfo.m_beauty(x, y);
            else
                filteredImage(x, y) /= weight_sum;
        }
    }
    return filteredImage;
}

void Denoiser::Init(const FrameInfo &frameInfo, const Buffer2D<Float3> &filteredColor) {
    m_accColor.Copy(filteredColor);
    int height = m_accColor.m_height;
    int width = m_accColor.m_width;
    m_misc = CreateBuffer2D<Float3>(width, height);
    m_valid = CreateBuffer2D<bool>(width, height);
}

void Denoiser::Maintain(const FrameInfo &frameInfo) { m_preFrameInfo = frameInfo; }

Buffer2D<Float3> Denoiser::ProcessFrame(const FrameInfo &frameInfo) {
    // Filter current frame
    Buffer2D<Float3> filteredColor;
    filteredColor = Filter(frameInfo);

    // Reproject previous frame color to current
    if (m_useTemportal) {
        Reprojection(frameInfo);
        TemporalAccumulation(filteredColor);
    } else {
        Init(frameInfo, filteredColor);
    }

    // Maintain
    Maintain(frameInfo);
    if (!m_useTemportal) {
        m_useTemportal = true;
    }
    return m_accColor;
}
